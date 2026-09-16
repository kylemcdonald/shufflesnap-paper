// Driver for evaluating BSP-OT (Genest et al., SIGGRAPH Asia 2025) on the
// point-to-grid assignment problem. Uses the official header-only release:
//   https://github.com/baptiste-genest/BSP-OT   (BSP-OT_header_only.h)
//
// Reads n 2D points (float64 binary), builds the same margin-inset grid as
// shufflesnap, computes a bijective matching with computeGaussianBSPOT (nb_plans
// randomized BSP plans merged cost-decreasingly), and reports the assignment
// cost and wall time. Requires n == rows * cols (bijective matching).
//
// Build (paths to Eigen and the BSP-OT header):
//   c++ -O3 -std=c++20 -fopenmp -I <eigen> -I <bspot-dir> \
//       solvers/cpp/bspot_grid.cpp -o solvers/cpp/bspot_grid

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "BSP-OT_header_only.h"

using namespace BSPOT;

int main(int argc, char** argv) {
    std::string points_path;
    std::string out_path;
    int rows = 0, cols = 0, nb_plans = 16;
    double margin = 0.03;
    unsigned seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (arg == "--points") points_path = next();
        else if (arg == "--rows") rows = std::stoi(next());
        else if (arg == "--cols") cols = std::stoi(next());
        else if (arg == "--plans") nb_plans = std::stoi(next());
        else if (arg == "--margin") margin = std::stod(next());
        else if (arg == "--seed") seed = std::stoul(next());
        else if (arg == "--assignment-out") out_path = next();
        else { std::cerr << "unknown arg " << arg << "\n"; return 1; }
    }
    const long n = static_cast<long>(rows) * cols;
    if (points_path.empty() || n <= 0) {
        std::cerr << "usage: bspot_grid --points FILE --rows R --cols C [--plans K]\n";
        return 1;
    }

    std::ifstream in(points_path, std::ios::binary);
    std::vector<double> buffer(2 * n);
    in.read(reinterpret_cast<char*>(buffer.data()), buffer.size() * sizeof(double));
    if (!in) { std::cerr << "failed to read " << 2 * n << " doubles\n"; return 1; }

    Points<2> A(2, n), B(2, n);
    for (long i = 0; i < n; ++i) {
        A(0, i) = buffer[2 * i];
        A(1, i) = buffer[2 * i + 1];
    }
    const double step_x = cols > 1 ? (1.0 - 2 * margin) / (cols - 1) : 0.0;
    const double step_y = rows > 1 ? (1.0 - 2 * margin) / (rows - 1) : 0.0;
    for (long r = 0; r < rows; ++r)
        for (long c = 0; c < cols; ++c) {
            B(0, r * cols + c) = cols > 1 ? margin + c * step_x : 0.5;
            B(1, r * cols + c) = rows > 1 ? margin + r * step_y : 0.5;
        }

    cost_function cost = [&](size_t i, size_t j) {
        return (A.col(i) - B.col(j)).squaredNorm();
    };

    srand(seed);
    const auto start = std::chrono::steady_clock::now();
    BijectiveMatching T = computeGaussianBSPOT<2>(A, B, nb_plans, cost);
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

    double total = 0.0;
    for (long i = 0; i < n; ++i) total += cost(i, T[i]);

    std::printf("n=%ld\nplans=%d\nsolve_time_s=%.9f\nassignment_cost=%.17g\nbijective=%d\n",
                n, nb_plans, elapsed, total, int(T.checkBijectivity()));

    if (!out_path.empty()) {
        std::ofstream out(out_path, std::ios::binary);
        std::vector<std::int64_t> assignment(n);
        for (long i = 0; i < n; ++i) assignment[i] = static_cast<std::int64_t>(T[i]);
        out.write(reinterpret_cast<const char*>(assignment.data()), n * sizeof(std::int64_t));
    }
    return 0;
}
