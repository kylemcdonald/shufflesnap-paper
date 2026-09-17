// Memory-lean single-plan BSP-OT driver for billion-point point-to-grid runs.
//
// The official BSP-OT release (BSP-OT_header_only.h, Genest et al. 2025;
// driver bspot_grid.cpp) cannot run n = 1e9 in 62 GiB: each BijectiveMatching
// stores a plan plus its inverse (8 GB), the projection records are 16-byte
// (double, int) pairs (32 GB for both sides), computeMoments copies each
// left slice into a thread-local buffer (up to 8 GB), MergePlans materializes
// per-point cost vectors (8 GB each), and the default 16 plans are all held
// at once. This file is a self-contained port of exactly one code path --
// BijectiveBSPMatching::computeGaussianMatching, i.e. what
// computeGaussianBSPOT(A, B, /*nb_plans=*/1, cost) computes, whose MergePlans
// step is a no-op for a single plan -- with the constants reduced:
//
//   - projection records are packed {float dot; int32 id} (8 bytes), so the
//     nth_element partitions order by float instead of double projections
//     (split sizes are unaffected: the pivot is positional);
//   - slice moments are streamed with no slice copy (same two-pass
//     mean-then-covariance arithmetic as the original computeMoments);
//   - the plan's inverse is never built (a bitset checks bijectivity);
//   - Eigen is replaced by closed-form 2x2 symmetric eigendecomposition
//     (the original solves only 2x2 systems here); eigenvalues are floored
//     at 1e-30 * trace before the inverse square root;
//   - the RNG is an mt19937 seeded by --seed for reproducibility (the
//     original seeds from std::random_device, so runs are unrepeatable);
//   - recursion is serial, as in the original (its plan loop parallelizes
//     across plans, which does not apply to a single plan).
//
// Peak memory at n = 1e9: points 16 GB + grid 16 GB + records 16 GB = 48 GiB.
//
// Faithfully preserved quirks: the base case (gap < 50) draws a Gaussian
// direction, projects out the previous direction, and then calls
// .normalized() without using the result -- the direction stays unnormalized
// in the original, so it stays unnormalized here (ordering is scale
// invariant).
//
// Build:
//   c++ -O3 -std=c++20 solvers/cpp/bspot_grid_giga.cpp -o solvers/cpp/bspot_grid_giga

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

std::mt19937 rng;

struct Vec2 {
    double x = 0.0, y = 0.0;
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double squaredNorm() const { return dot(*this); }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2 operator/(double s) const { return {x / s, y / s}; }
};

// symmetric 2x2 matrix [[a, b], [b, c]]
struct Sym2 {
    double a = 0.0, b = 0.0, c = 0.0;
    Sym2 operator+(const Sym2& o) const { return {a + o.a, b + o.b, c + o.c}; }
    Sym2 operator-(const Sym2& o) const { return {a - o.a, b - o.b, c - o.c}; }
    Sym2 operator*(double s) const { return {a * s, b * s, c * s}; }
    Sym2 operator/(double s) const { return {a / s, b / s, c / s}; }
};

Sym2 outer(const Vec2& v) { return {v.x * v.x, v.x * v.y, v.y * v.y}; }

struct Moments {
    Vec2 mean;
    Sym2 cov;
};

struct EigenPair {
    double l1 = 0.0, l2 = 0.0;  // l1 >= l2
    Vec2 v1, v2;                // unit eigenvectors
};

EigenPair eigen2x2(const Sym2& s) {
    EigenPair e;
    const double half_tr = 0.5 * (s.a + s.c);
    const double disc = std::sqrt(std::max(0.0, 0.25 * (s.a - s.c) * (s.a - s.c) + s.b * s.b));
    e.l1 = half_tr + disc;
    e.l2 = half_tr - disc;
    if (std::abs(s.b) > 1e-300) {
        e.v1 = {e.l1 - s.c, s.b};
        e.v2 = {e.l2 - s.c, s.b};
    } else if (s.a >= s.c) {
        e.v1 = {1.0, 0.0};
        e.v2 = {0.0, 1.0};
    } else {
        e.v1 = {0.0, 1.0};
        e.v2 = {1.0, 0.0};
    }
    const double n1 = std::sqrt(e.v1.squaredNorm());
    const double n2 = std::sqrt(e.v2.squaredNorm());
    if (n1 > 0.0) e.v1 = e.v1 / n1;
    if (n2 > 0.0) e.v2 = e.v2 / n2;
    return e;
}

// V f(diag) V^T for the symmetric matrix with eigenpair e
Sym2 apply_spectral(const EigenPair& e, double f1, double f2) {
    return outer(e.v1) * f1 + outer(e.v2) * f2;
}

// R * S * R for symmetric R, S: the intermediate R * S is not symmetric, so
// it is carried as a full 2x2 product; the triple product is symmetric and
// only its off-diagonal is averaged against floating-point noise.
Sym2 sandwich(const Sym2& r, const Sym2& s) {
    const double p00 = r.a * s.a + r.b * s.b;
    const double p01 = r.a * s.b + r.b * s.c;
    const double p10 = r.b * s.a + r.c * s.b;
    const double p11 = r.b * s.b + r.c * s.c;
    const double q00 = p00 * r.a + p01 * r.b;
    const double q01 = p00 * r.b + p01 * r.c;
    const double q10 = p10 * r.a + p11 * r.b;
    const double q11 = p10 * r.b + p11 * r.c;
    return {q00, 0.5 * (q01 + q10), q11};
}

double eigen_floor(const Sym2& s) { return 1e-30 * std::max(1e-300, s.a + s.c); }

// A^{-1/2} (A^{1/2} B A^{1/2})^{1/2} A^{-1/2}, as in W2GaussianTransportMap
Sym2 w2_transport_map(const Sym2& A, const Sym2& B) {
    EigenPair ea = eigen2x2(A);
    const double fa = eigen_floor(A);
    const double r1 = std::sqrt(std::max(fa, ea.l1));
    const double r2 = std::sqrt(std::max(fa, ea.l2));
    Sym2 root = apply_spectral(ea, r1, r2);
    Sym2 inv_root = apply_spectral(ea, 1.0 / r1, 1.0 / r2);
    Sym2 C = sandwich(root, B);
    EigenPair ec = eigen2x2(C);
    const double fc = eigen_floor(C);
    C = apply_spectral(ec, std::sqrt(std::max(fc, ec.l1)), std::sqrt(std::max(fc, ec.l2)));
    return sandwich(inv_root, C);
}

Vec2 sample_unit_gaussian() {
    std::normal_distribution<double> dist{0.0, 1.0};
    return {dist(rng), dist(rng)};
}

int randint(int a, int b) {
    std::uniform_int_distribution<int> dist(a, b);
    return dist(rng);
}

struct DotId {
    float dot;
    std::int32_t id;
    bool operator<(const DotId& o) const { return dot < o.dot; }
};
static_assert(sizeof(DotId) == 8, "DotId must stay packed");

using Records = std::vector<DotId>;

class GigaBSP {
public:
    GigaBSP(const double* A, const double* B, std::int64_t n) : A_(A), B_(B), n_(n) {}

    std::vector<std::int32_t> compute_gaussian_matching() {
        Records idA(n_), idB(n_);
        for (std::int64_t i = 0; i < n_; ++i) {
            idA[i].id = static_cast<std::int32_t>(i);
            idB[i].id = static_cast<std::int32_t>(i);
        }
        const Moments ma = global_moments(A_);
        const Moments mb = global_moments(B_);
        gaussian_partial(idA, idB, 0, static_cast<std::int64_t>(n_), ma, mb);

        std::vector<std::int32_t> plan(n_);
        for (std::int64_t i = 0; i < n_; ++i) {
            plan[static_cast<std::size_t>(idA[i].id)] = idB[i].id;
        }
        return plan;
    }

private:
    const double* A_;
    const double* B_;
    std::int64_t n_;

    static Vec2 col(const double* pts, std::int64_t i) { return {pts[2 * i], pts[2 * i + 1]}; }

    Moments global_moments(const double* pts) const {
        Vec2 mean;
        for (std::int64_t i = 0; i < n_; ++i) mean = mean + col(pts, i);
        mean = mean / static_cast<double>(n_);
        Sym2 cov;
        for (std::int64_t i = 0; i < n_; ++i) cov = cov + outer(col(pts, i) - mean);
        return {mean, cov / static_cast<double>(n_)};
    }

    // streaming equivalent of computeMoments over id[beg, end)
    Moments slice_moments(const double* pts, const Records& id,
                          std::int64_t beg, std::int64_t end) const {
        Vec2 mean;
        for (std::int64_t i = beg; i < end; ++i) mean = mean + col(pts, id[i].id);
        mean = mean / static_cast<double>(end - beg);
        Sym2 cov;
        for (std::int64_t i = beg; i < end; ++i) {
            cov = cov + outer(col(pts, id[i].id) - mean) / static_cast<double>(end - beg);
        }
        return {mean, cov};
    }

    // decomposeMoments: exact left moments, right derived algebraically
    std::pair<Moments, Moments> decompose_moments(const double* pts, const Moments& m,
                                                  const Records& id, std::int64_t beg,
                                                  std::int64_t end, std::int64_t pivot) const {
        const double alpha = static_cast<double>(pivot - beg) / static_cast<double>(end - beg);
        const double beta = 1.0 - alpha;
        const Moments left = slice_moments(pts, id, beg, pivot);
        const Vec2 mr = (m.mean - left.mean * alpha) / beta;
        const Sym2 dl = outer(m.mean - left.mean);
        const Sym2 dr = outer(m.mean - mr);
        const Sym2 cr = (m.cov - (left.cov + dl) * alpha) / beta - dr;
        return {left, {mr, cr}};
    }

    void project_and_split(Records& idA, Records& idB, std::int64_t beg, std::int64_t end,
                           std::int64_t pivot, const Vec2& d) {
        for (std::int64_t i = beg; i < end; ++i) {
            idA[i].dot = static_cast<float>(d.dot(col(A_, idA[i].id)));
            idB[i].dot = static_cast<float>(d.dot(col(B_, idB[i].id)));
        }
        std::nth_element(idA.begin() + beg, idA.begin() + pivot, idA.begin() + end);
        std::nth_element(idB.begin() + beg, idB.begin() + pivot, idB.begin() + end);
    }

    void gaussian_partial(Records& idA, Records& idB, std::int64_t beg, std::int64_t end,
                          const Moments& ma, const Moments& mb) {
        const std::int64_t gap = end - beg;
        if (gap <= 1) return;
        if (gap < 50) {
            partial_orthogonal(idA, idB, beg, end, sample_unit_gaussian());
            return;
        }
        const EigenPair et = eigen2x2(w2_transport_map(ma.cov, mb.cov));
        const Vec2 d = randint(0, 1) == 0 ? et.v1 : et.v2;
        const std::int64_t pivot = beg + gap / 2;  // random_pivot=false path
        project_and_split(idA, idB, beg, end, pivot, d);
        const auto sma = decompose_moments(A_, ma, idA, beg, end, pivot);
        const auto smb = decompose_moments(B_, mb, idB, beg, end, pivot);
        gaussian_partial(idA, idB, beg, pivot, sma.first, smb.first);
        gaussian_partial(idA, idB, pivot, end, sma.second, smb.second);
    }

    void partial_orthogonal(Records& idA, Records& idB, std::int64_t beg, std::int64_t end,
                            const Vec2& prev) {
        const std::int64_t gap = end - beg;
        if (gap <= 1) return;
        const std::int64_t pivot = beg + gap / 2;
        Vec2 d = sample_unit_gaussian();
        d = d - prev * (d.dot(prev) / prev.squaredNorm());
        // the original calls d.normalized() and discards the result; ordering
        // by projection is scale invariant, so d is left unnormalized here too
        project_and_split(idA, idB, beg, end, pivot, d);
        partial_orthogonal(idA, idB, beg, pivot, d);
        partial_orthogonal(idA, idB, pivot, end, d);
    }
};

bool read_points(const std::string& path, double* dst, std::int64_t count) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    constexpr std::int64_t chunk = std::int64_t{1} << 27;  // 1 GiB of doubles
    std::int64_t done = 0;
    while (done < count) {
        const std::int64_t now = std::min(chunk, count - done);
        in.read(reinterpret_cast<char*>(dst + done), now * static_cast<std::int64_t>(sizeof(double)));
        if (!in) return false;
        done += now;
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string points_path;
    std::string out_path;
    std::int64_t rows = 0, cols = 0;
    unsigned seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--points") points_path = next();
        else if (arg == "--rows") rows = std::stoll(next());
        else if (arg == "--cols") cols = std::stoll(next());
        else if (arg == "--seed") seed = std::stoul(next());
        else if (arg == "--assignment-out") out_path = next();
        else { std::cerr << "unknown arg " << arg << "\n"; return 1; }
    }
    const std::int64_t n = rows * cols;
    if (points_path.empty() || n <= 0 || n > INT32_MAX) {
        std::cerr << "usage: bspot_grid_giga --points FILE --rows R --cols C"
                  << " [--seed S] [--assignment-out FILE]\n"
                  << "(rows * cols must be positive and fit in int32)\n";
        return 1;
    }
    rng.seed(seed);

    std::vector<double> A(2 * n);
    if (!read_points(points_path, A.data(), 2 * n)) {
        std::cerr << "failed to read " << 2 * n << " doubles from " << points_path << "\n";
        return 1;
    }

    const double step_x = cols > 1 ? 1.0 / static_cast<double>(cols - 1) : 0.0;
    const double step_y = rows > 1 ? 1.0 / static_cast<double>(rows - 1) : 0.0;
    const auto target_x = [&](std::int64_t j) {
        return cols > 1 ? static_cast<double>(j % cols) * step_x : 0.5;
    };
    const auto target_y = [&](std::int64_t j) {
        return rows > 1 ? static_cast<double>(j / cols) * step_y : 0.5;
    };

    std::vector<std::int32_t> plan;
    double elapsed = 0.0;
    {
        std::vector<double> B(2 * n);
        for (std::int64_t j = 0; j < n; ++j) {
            B[2 * j] = target_x(j);
            B[2 * j + 1] = target_y(j);
        }
        const auto start = std::chrono::steady_clock::now();
        GigaBSP bsp(A.data(), B.data(), n);
        plan = bsp.compute_gaussian_matching();
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    std::vector<bool> hit(static_cast<std::size_t>(n), false);
    bool bijective = true;
    double total = 0.0;
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t j = plan[i];
        if (j < 0 || j >= n || hit[static_cast<std::size_t>(j)]) bijective = false;
        else hit[static_cast<std::size_t>(j)] = true;
        const double dx = A[2 * i] - target_x(j);
        const double dy = A[2 * i + 1] - target_y(j);
        total += dx * dx + dy * dy;
    }

    std::printf("n=%lld\nplans=1\nsolve_time_s=%.9f\nassignment_cost=%.17g\nbijective=%d\n",
                static_cast<long long>(n), elapsed, total, static_cast<int>(bijective));

    if (!out_path.empty()) {
        std::ofstream out(out_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(plan.data()),
                  static_cast<std::streamsize>(plan.size() * sizeof(std::int32_t)));
        if (!out) { std::cerr << "failed to write " << out_path << "\n"; return 1; }
    }
    return 0;
}
