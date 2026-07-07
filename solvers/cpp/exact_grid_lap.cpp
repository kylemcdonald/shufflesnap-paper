#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

struct Point {
    double x = 0.0;
    double y = 0.0;
};

struct Options {
    std::string points_path;
    std::string assignment_out_path;
    std::string solver = "jv";
    std::string cost_mode = "jit";
    int width = 0;
    int height = 0;
    double margin = 0.03;
    int threads = 0;
    int repeat = 1;
    int warmup = 0;
    bool debug_auction = false;
    int debug_round_every = 0;
};

struct SolveStats {
    double build_time_s = 0.0;
    double solve_time_s = 0.0;
    double total_time_s = 0.0;
    double assignment_cost = 0.0;
    std::size_t n = 0;
    std::size_t cost_evaluations = 0;
    std::vector<std::int64_t> assignment;
};

struct AssignmentResult {
    std::vector<std::int64_t> assignment;
    double total_cost = 0.0;
    std::size_t cost_evaluations = 0;
};

double now_seconds() {
    using clock = std::chrono::steady_clock;
    const auto now = clock::now().time_since_epoch();
    return std::chrono::duration_cast<std::chrono::duration<double>>(now).count();
}

int resolve_threads(int requested, std::size_t max_tasks = 0) {
    int thread_count = requested;
    if (thread_count <= 0) {
        const unsigned int detected = std::thread::hardware_concurrency();
        thread_count = detected == 0 ? 1 : static_cast<int>(detected);
    }
    thread_count = std::max(1, thread_count);
    if (max_tasks > 0) {
        thread_count = std::min<int>(thread_count, static_cast<int>(max_tasks));
    }
    return thread_count;
}

double median_of(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    const auto mid = values.begin() + static_cast<std::ptrdiff_t>(values.size() / 2);
    std::nth_element(values.begin(), mid, values.end());
    if ((values.size() % 2) == 1) {
        return *mid;
    }
    const double upper = *mid;
    const auto lower_mid = values.begin() + static_cast<std::ptrdiff_t>((values.size() / 2) - 1);
    std::nth_element(values.begin(), lower_mid, values.end());
    return 0.5 * (*lower_mid + upper);
}

Options parse_args(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto require_value = [&](const char* name) -> std::string {
            if (index + 1 >= argc) {
                throw std::runtime_error(std::string("missing value for ") + name);
            }
            ++index;
            return std::string(argv[index]);
        };
        if (arg == "--points") {
            options.points_path = require_value("--points");
        } else if (arg == "--assignment-out") {
            options.assignment_out_path = require_value("--assignment-out");
        } else if (arg == "--width") {
            options.width = std::stoi(require_value("--width"));
        } else if (arg == "--height") {
            options.height = std::stoi(require_value("--height"));
        } else if (arg == "--solver") {
            options.solver = require_value("--solver");
        } else if (arg == "--cost-mode") {
            options.cost_mode = require_value("--cost-mode");
        } else if (arg == "--margin") {
            options.margin = std::stod(require_value("--margin"));
        } else if (arg == "--threads") {
            options.threads = std::stoi(require_value("--threads"));
        } else if (arg == "--repeat") {
            options.repeat = std::stoi(require_value("--repeat"));
        } else if (arg == "--warmup") {
            options.warmup = std::stoi(require_value("--warmup"));
        } else if (arg == "--debug-auction") {
            options.debug_auction = true;
        } else if (arg == "--debug-round-every") {
            options.debug_round_every = std::stoi(require_value("--debug-round-every"));
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: exact_grid_lap --points PATH --width W --height H "
                << "--solver jv|auction --cost-mode dense|jit "
                << "[--margin 0.03] [--threads N] [--repeat N] [--warmup N] "
                << "[--debug-auction] [--debug-round-every N] "
                << "[--assignment-out PATH]\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }
    if (options.points_path.empty()) {
        throw std::runtime_error("--points is required");
    }
    if (options.width <= 0 || options.height <= 0) {
        throw std::runtime_error("--width and --height must be positive");
    }
    if (options.solver != "jv" && options.solver != "auction") {
        throw std::runtime_error("--solver must be one of: jv, auction");
    }
    if (options.cost_mode != "dense" && options.cost_mode != "jit") {
        throw std::runtime_error("--cost-mode must be one of: dense, jit");
    }
    if (options.repeat <= 0) {
        throw std::runtime_error("--repeat must be positive");
    }
    if (options.warmup < 0) {
        throw std::runtime_error("--warmup must be non-negative");
    }
    if (options.debug_round_every < 0) {
        throw std::runtime_error("--debug-round-every must be non-negative");
    }
    return options;
}

std::vector<Point> load_points(const std::string& path, std::size_t expected_n) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open points file: " + path);
    }
    input.seekg(0, std::ios::end);
    const auto bytes = static_cast<std::size_t>(input.tellg());
    input.seekg(0, std::ios::beg);
    if (bytes != expected_n * sizeof(Point)) {
        throw std::runtime_error("unexpected points file size");
    }
    std::vector<Point> points(expected_n);
    input.read(reinterpret_cast<char*>(points.data()), static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error("failed to read points file");
    }
    return points;
}

void write_assignment(const std::string& path, const std::vector<std::int64_t>& assignment) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        throw std::runtime_error("failed to open assignment output path: " + path);
    }
    output.write(
        reinterpret_cast<const char*>(assignment.data()),
        static_cast<std::streamsize>(assignment.size() * sizeof(std::int64_t))
    );
    if (!output) {
        throw std::runtime_error("failed to write assignment output path: " + path);
    }
}

struct GridGeometry {
    int width = 0;
    int height = 0;
    std::vector<double> xs;
    std::vector<double> ys;

    explicit GridGeometry(int grid_width, int grid_height, double margin)
        : width(grid_width), height(grid_height), xs(static_cast<std::size_t>(grid_width)), ys(static_cast<std::size_t>(grid_height)) {
        if (width == 1) {
            xs[0] = 0.5;
        } else {
            for (int col = 0; col < width; ++col) {
                xs[static_cast<std::size_t>(col)] =
                    margin + (1.0 - (2.0 * margin)) * static_cast<double>(col) / static_cast<double>(width - 1);
            }
        }
        if (height == 1) {
            ys[0] = 0.5;
        } else {
            for (int row = 0; row < height; ++row) {
                ys[static_cast<std::size_t>(row)] =
                    margin + (1.0 - (2.0 * margin)) * static_cast<double>(row) / static_cast<double>(height - 1);
            }
        }
    }

    inline double target_x(std::size_t target) const {
        return xs[target % static_cast<std::size_t>(width)];
    }

    inline double target_y(std::size_t target) const {
        return ys[target / static_cast<std::size_t>(width)];
    }
};

double squared_distance(const Point& point, double target_x, double target_y) {
    const double dx = point.x - target_x;
    const double dy = point.y - target_y;
    return (dx * dx) + (dy * dy);
}

std::vector<double> build_dense_cost_matrix(
    const std::vector<Point>& points,
    const GridGeometry& grid,
    int threads
) {
    const std::size_t n = points.size();
    std::vector<double> cost(n * n, 0.0);
    const int thread_count = resolve_threads(threads, n);
    std::vector<std::thread> workers;
    workers.reserve(static_cast<std::size_t>(thread_count));

    auto fill_range = [&](std::size_t row_begin, std::size_t row_end) {
        for (std::size_t row = row_begin; row < row_end; ++row) {
            const Point point = points[row];
            const std::size_t row_offset = row * n;
            for (std::size_t col = 0; col < n; ++col) {
                cost[row_offset + col] = squared_distance(point, grid.target_x(col), grid.target_y(col));
            }
        }
    };

    const std::size_t chunk = (n + static_cast<std::size_t>(thread_count) - 1) / static_cast<std::size_t>(thread_count);
    std::size_t row_begin = 0;
    for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
        const std::size_t row_end = std::min<std::size_t>(n, row_begin + chunk);
        if (row_begin >= row_end) {
            break;
        }
        workers.emplace_back(fill_range, row_begin, row_end);
        row_begin = row_end;
    }
    for (auto& worker : workers) {
        worker.join();
    }
    return cost;
}

template <typename CostAccessor>
AssignmentResult solve_square_jv(std::size_t n, const CostAccessor& cost_accessor) {
    if (n == 0) {
        return AssignmentResult{};
    }

    std::vector<double> u(n, 0.0);
    std::vector<double> v(n, 0.0);
    std::vector<double> shortest(n, kInfinity);
    std::vector<std::int64_t> path(n, -1);
    std::vector<std::int64_t> col4row(n, -1);
    std::vector<std::int64_t> row4col(n, -1);
    std::vector<std::size_t> remaining(n, 0);
    std::vector<unsigned char> sr(n, 0);
    std::vector<unsigned char> sc(n, 0);
    std::size_t evaluations = 0;

    for (std::size_t cur_row = 0; cur_row < n; ++cur_row) {
        double min_val = 0.0;
        std::size_t i = cur_row;
        std::size_t num_remaining = n;

        for (std::size_t it = 0; it < n; ++it) {
            remaining[it] = n - it - 1;
            shortest[it] = kInfinity;
            path[it] = -1;
            sr[it] = 0;
            sc[it] = 0;
        }

        std::size_t sink = n;
        while (sink == n) {
            std::size_t index = n;
            double lowest = kInfinity;
            sr[i] = 1;

            for (std::size_t it = 0; it < num_remaining; ++it) {
                const std::size_t j = remaining[it];
                const double reduced_cost = min_val + cost_accessor(i, j) - u[i] - v[j];
                ++evaluations;
                if (reduced_cost < shortest[j]) {
                    path[j] = static_cast<std::int64_t>(i);
                    shortest[j] = reduced_cost;
                }
                if (shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == -1)) {
                    lowest = shortest[j];
                    index = it;
                }
            }

            min_val = lowest;
            if (!std::isfinite(min_val)) {
                throw std::runtime_error("infeasible assignment");
            }

            const std::size_t j = remaining[index];
            if (row4col[j] == -1) {
                sink = j;
            } else {
                i = static_cast<std::size_t>(row4col[j]);
            }
            sc[j] = 1;
            --num_remaining;
            remaining[index] = remaining[num_remaining];
        }

        u[cur_row] += min_val;
        for (std::size_t row = 0; row < n; ++row) {
            if (sr[row] && row != cur_row) {
                const std::size_t col = static_cast<std::size_t>(col4row[row]);
                u[row] += min_val - shortest[col];
            }
        }
        for (std::size_t col = 0; col < n; ++col) {
            if (sc[col]) {
                v[col] -= min_val - shortest[col];
            }
        }

        std::size_t j = sink;
        while (true) {
            const auto row = static_cast<std::size_t>(path[j]);
            row4col[j] = static_cast<std::int64_t>(row);
            const auto previous_j = col4row[row];
            col4row[row] = static_cast<std::int64_t>(j);
            if (row == cur_row) {
                break;
            }
            j = static_cast<std::size_t>(previous_j);
        }
    }

    AssignmentResult result;
    result.assignment.resize(n);
    result.cost_evaluations = evaluations;
    for (std::size_t row = 0; row < n; ++row) {
        const std::size_t col = static_cast<std::size_t>(col4row[row]);
        result.assignment[row] = static_cast<std::int64_t>(col);
        result.total_cost += cost_accessor(row, col);
        ++result.cost_evaluations;
    }
    return result;
}

template <typename CostAccessor>
AssignmentResult solve_square_auction(
    std::size_t n,
    const CostAccessor& cost_accessor,
    int threads,
    bool debug,
    int debug_round_every
) {
    if (n == 0) {
        return AssignmentResult{};
    }

    double min_cost = kInfinity;
    double max_cost = -kInfinity;
    std::size_t evaluations = 0;
    for (std::size_t row = 0; row < n; ++row) {
        for (std::size_t col = 0; col < n; ++col) {
            const double value = cost_accessor(row, col);
            ++evaluations;
            min_cost = std::min(min_cost, value);
            max_cost = std::max(max_cost, value);
        }
    }

    std::vector<double> prices(n, 0.0);
    std::vector<std::size_t> owner(n, n);
    std::vector<std::size_t> assignment(n, n);
    std::vector<std::size_t> unassigned(n, 0);
    std::vector<std::size_t> next_unassigned;
    next_unassigned.reserve(n);

    const double span = std::max(std::abs(max_cost - min_cost), 1.0);
    const double user_final_epsilon = 1e-12 / std::max<double>(1.0, static_cast<double>(n));
    const double cost_scale = std::max({1.0, std::abs(min_cost), std::abs(max_cost)});
    const double representable_final_epsilon = std::nextafter(cost_scale, kInfinity) - cost_scale;
    const double final_epsilon = std::max(user_final_epsilon, representable_final_epsilon);
    double epsilon = std::max(span / 4.0, final_epsilon);
    const int thread_count = resolve_threads(threads, n);

    struct Bid {
        std::size_t bidder = 0;
        std::size_t object = 0;
        double amount = 0.0;
    };

    std::vector<Bid> bids(n);
    std::vector<double> best_amount(n, -kInfinity);
    std::vector<std::size_t> best_bidder(n, n);

    int phase_index = 0;
    while (epsilon >= final_epsilon) {
        const double phase_start = now_seconds();
        std::size_t phase_rounds = 0;
        std::size_t phase_evaluations = 0;
        std::size_t phase_reopened = 0;
        std::size_t unassigned_count = 0;
        if (phase_index == 0) {
            std::fill(owner.begin(), owner.end(), n);
            std::fill(assignment.begin(), assignment.end(), n);
            for (std::size_t index = 0; index < n; ++index) {
                unassigned[index] = index;
            }
            unassigned_count = n;
            phase_reopened = n;
        } else {
            for (std::size_t bidder = 0; bidder < n; ++bidder) {
                const std::size_t assigned_object = assignment[bidder];
                if (assigned_object >= n || owner[assigned_object] != bidder) {
                    throw std::runtime_error("auction warm start has inconsistent assignment");
                }

                double best_value = kInfinity;
                double assigned_value = kInfinity;
                for (std::size_t object = 0; object < n; ++object) {
                    const double value = cost_accessor(bidder, object) + prices[object];
                    ++evaluations;
                    ++phase_evaluations;
                    if (value < best_value) {
                        best_value = value;
                    }
                    if (object == assigned_object) {
                        assigned_value = value;
                    }
                }

                if (!std::isfinite(assigned_value) || !std::isfinite(best_value)) {
                    throw std::runtime_error("infeasible assignment");
                }
                if (assigned_value > best_value + epsilon) {
                    assignment[bidder] = n;
                    owner[assigned_object] = n;
                    unassigned[unassigned_count] = bidder;
                    ++unassigned_count;
                    ++phase_reopened;
                }
            }
        }
        if (debug) {
            std::cerr
                << "[auction] phase_start"
                << " phase=" << phase_index
                << " epsilon=" << std::setprecision(17) << epsilon
                << " final_epsilon=" << final_epsilon
                << " reopened=" << phase_reopened
                << " unassigned=" << unassigned_count
                << "\n";
        }

        while (unassigned_count > 0) {
            ++phase_rounds;
            const std::size_t round_start_unassigned = unassigned_count;
            std::vector<std::size_t> local_evals(static_cast<std::size_t>(thread_count), 0);
            auto evaluate_range = [&](int thread_index, std::size_t begin, std::size_t end) {
                std::size_t evals = 0;
                for (std::size_t idx = begin; idx < end; ++idx) {
                    const std::size_t bidder = unassigned[idx];
                    std::size_t best_object = 0;
                    double best_value = kInfinity;
                    double second_value = kInfinity;
                    for (std::size_t object = 0; object < n; ++object) {
                        const double value = cost_accessor(bidder, object) + prices[object];
                        ++evals;
                        if (value < best_value) {
                            second_value = best_value;
                            best_value = value;
                            best_object = object;
                        } else if (value < second_value) {
                            second_value = value;
                        }
                    }
                    if (!std::isfinite(best_value)) {
                        throw std::runtime_error("infeasible assignment");
                    }
                    if (!std::isfinite(second_value)) {
                        second_value = best_value;
                    }
                    bids[idx] = Bid{
                        bidder,
                        best_object,
                        std::max(0.0, second_value - best_value) + epsilon,
                    };
                }
                local_evals[static_cast<std::size_t>(thread_index)] = evals;
            };

            const std::size_t chunk =
                (unassigned_count + static_cast<std::size_t>(thread_count) - 1) / static_cast<std::size_t>(thread_count);
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(thread_count));
            for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
                const std::size_t begin = static_cast<std::size_t>(thread_index) * chunk;
                const std::size_t end = std::min<std::size_t>(unassigned_count, begin + chunk);
                if (begin >= end) {
                    break;
                }
                workers.emplace_back(evaluate_range, thread_index, begin, end);
            }
            for (auto& worker : workers) {
                worker.join();
            }
            for (const std::size_t evals : local_evals) {
                evaluations += evals;
                phase_evaluations += evals;
            }

            std::fill(best_amount.begin(), best_amount.end(), -kInfinity);
            std::fill(best_bidder.begin(), best_bidder.end(), n);
            next_unassigned.clear();

            for (std::size_t idx = 0; idx < unassigned_count; ++idx) {
                const Bid& bid = bids[idx];
                if (bid.amount > best_amount[bid.object]) {
                    best_amount[bid.object] = bid.amount;
                    best_bidder[bid.object] = bid.bidder;
                }
            }

            std::size_t winner_count = 0;
            std::size_t ineffective_price_updates = 0;
            std::size_t forced_nextafter_updates = 0;
            std::size_t reassignment_count = 0;
            double min_winning_bid = kInfinity;
            double max_winning_bid = 0.0;
            double sum_winning_bid = 0.0;
            for (std::size_t idx = 0; idx < unassigned_count; ++idx) {
                const Bid& bid = bids[idx];
                if (best_bidder[bid.object] != bid.bidder) {
                    next_unassigned.push_back(bid.bidder);
                }
            }

            for (std::size_t object = 0; object < n; ++object) {
                const std::size_t bidder = best_bidder[object];
                if (bidder == n) {
                    continue;
                }
                ++winner_count;
                const double old_price = prices[object];
                const double bid_amount = best_amount[object];
                min_winning_bid = std::min(min_winning_bid, bid_amount);
                max_winning_bid = std::max(max_winning_bid, bid_amount);
                sum_winning_bid += bid_amount;
                double new_price = old_price + bid_amount;
                if (new_price == old_price && bid_amount > 0.0) {
                    new_price = std::nextafter(old_price, kInfinity);
                    ++forced_nextafter_updates;
                }
                prices[object] = new_price;
                if (prices[object] == old_price) {
                    ++ineffective_price_updates;
                }
                const std::size_t previous_owner = owner[object];
                owner[object] = bidder;
                assignment[bidder] = object;
                if (previous_owner != n) {
                    ++reassignment_count;
                    assignment[previous_owner] = n;
                    next_unassigned.push_back(previous_owner);
                }
            }

            unassigned_count = next_unassigned.size();
            for (std::size_t idx = 0; idx < unassigned_count; ++idx) {
                unassigned[idx] = next_unassigned[idx];
            }

            if (
                debug
                && (
                    debug_round_every > 0
                    && (
                        (phase_rounds % static_cast<std::size_t>(debug_round_every)) == 0
                        || ineffective_price_updates > 0
                        || unassigned_count <= 32
                    )
                )
            ) {
                double min_price = kInfinity;
                double max_price = -kInfinity;
                for (const double price : prices) {
                    min_price = std::min(min_price, price);
                    max_price = std::max(max_price, price);
                }
                const double avg_winning_bid = winner_count == 0 ? 0.0 : (sum_winning_bid / static_cast<double>(winner_count));
                std::cerr
                    << "[auction] round"
                    << " phase=" << phase_index
                    << " round=" << phase_rounds
                    << " unassigned_start=" << round_start_unassigned
                    << " unassigned_end=" << unassigned_count
                    << " winners=" << winner_count
                    << " reassignments=" << reassignment_count
                    << " ineffective_price_updates=" << ineffective_price_updates
                    << " forced_nextafter_updates=" << forced_nextafter_updates
                    << " avg_winning_bid=" << avg_winning_bid
                    << " min_winning_bid=" << (winner_count == 0 ? 0.0 : min_winning_bid)
                    << " max_winning_bid=" << max_winning_bid
                    << " min_price=" << min_price
                    << " max_price=" << max_price
                    << "\n";
            }
        }

        if (debug) {
            double min_price = kInfinity;
            double max_price = -kInfinity;
            for (const double price : prices) {
                min_price = std::min(min_price, price);
                max_price = std::max(max_price, price);
            }
            std::cerr
                << "[auction] phase_done"
                << " phase=" << phase_index
                << " epsilon=" << std::setprecision(17) << epsilon
                << " rounds=" << phase_rounds
                << " phase_evaluations=" << phase_evaluations
                << " elapsed_s=" << (now_seconds() - phase_start)
                << " min_price=" << min_price
                << " max_price=" << max_price
                << "\n";
        }

        if (epsilon == final_epsilon) {
            break;
        }
        epsilon = std::max(epsilon * 0.2, final_epsilon);
        ++phase_index;
    }

    AssignmentResult result;
    result.assignment.resize(n);
    result.cost_evaluations = evaluations;
    std::vector<unsigned char> seen(n, 0);
    for (std::size_t row = 0; row < n; ++row) {
        const std::size_t col = assignment[row];
        if (col >= n || seen[col]) {
            throw std::runtime_error("auction failed to produce a permutation");
        }
        seen[col] = 1;
        result.assignment[row] = static_cast<std::int64_t>(col);
        result.total_cost += cost_accessor(row, col);
        ++result.cost_evaluations;
    }
    return result;
}

template <typename Solver>
SolveStats run_solver(
    const std::vector<Point>& points,
    const GridGeometry& grid,
    const Options& options,
    const Solver& solver
) {
    const std::size_t n = points.size();
    const double total_start = now_seconds();
    AssignmentResult result;
    double build_time_s = 0.0;
    if (options.cost_mode == "dense") {
        const double build_start = now_seconds();
        std::vector<double> cost = build_dense_cost_matrix(points, grid, options.threads);
        build_time_s = now_seconds() - build_start;
        const auto dense_accessor = [&](std::size_t row, std::size_t col) -> double {
            return cost[(row * n) + col];
        };
        const double solve_start = now_seconds();
        result = solver(n, dense_accessor);
        const double solve_time_s = now_seconds() - solve_start;
        return SolveStats{
            build_time_s,
            solve_time_s,
            now_seconds() - total_start,
            result.total_cost,
            n,
            result.cost_evaluations,
            std::move(result.assignment),
        };
    }

    const auto jit_accessor = [&](std::size_t row, std::size_t col) -> double {
        return squared_distance(points[row], grid.target_x(col), grid.target_y(col));
    };
    const double solve_start = now_seconds();
    result = solver(n, jit_accessor);
    const double solve_time_s = now_seconds() - solve_start;
    return SolveStats{
        0.0,
        solve_time_s,
        now_seconds() - total_start,
        result.total_cost,
        n,
        result.cost_evaluations,
        std::move(result.assignment),
    };
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_args(argc, argv);
        const std::size_t n = static_cast<std::size_t>(options.width) * static_cast<std::size_t>(options.height);
        const std::vector<Point> points = load_points(options.points_path, n);
        const GridGeometry grid(options.width, options.height, options.margin);
        std::vector<double> build_times;
        std::vector<double> solve_times;
        std::vector<double> total_times;
        build_times.reserve(static_cast<std::size_t>(options.repeat));
        solve_times.reserve(static_cast<std::size_t>(options.repeat));
        total_times.reserve(static_cast<std::size_t>(options.repeat));

        SolveStats last_stats;
        for (int iteration = 0; iteration < options.warmup + options.repeat; ++iteration) {
            SolveStats stats;
            if (options.solver == "jv") {
                const auto solver = [&](std::size_t size, const auto& accessor) {
                    return solve_square_jv(size, accessor);
                };
                stats = run_solver(points, grid, options, solver);
            } else {
                const auto solver = [&](std::size_t size, const auto& accessor) {
                    return solve_square_auction(
                        size,
                        accessor,
                        options.threads,
                        options.debug_auction,
                        options.debug_round_every
                    );
                };
                stats = run_solver(points, grid, options, solver);
            }
            last_stats = stats;
            if (iteration >= options.warmup) {
                build_times.push_back(stats.build_time_s);
                solve_times.push_back(stats.solve_time_s);
                total_times.push_back(stats.total_time_s);
            }
        }
        const SolveStats stats{
            median_of(build_times),
            median_of(solve_times),
            median_of(total_times),
            last_stats.assignment_cost,
            last_stats.n,
            last_stats.cost_evaluations,
            std::move(last_stats.assignment),
        };

        if (!options.assignment_out_path.empty()) {
            write_assignment(options.assignment_out_path, stats.assignment);
        }

        std::cout << std::setprecision(17);
        std::cout << "solver=" << options.solver << "\n";
        std::cout << "cost_mode=" << options.cost_mode << "\n";
        std::cout << "n=" << stats.n << "\n";
        std::cout << "threads=" << resolve_threads(options.threads, stats.n) << "\n";
        std::cout << "repeat=" << options.repeat << "\n";
        std::cout << "warmup=" << options.warmup << "\n";
        std::cout << "build_time_s=" << stats.build_time_s << "\n";
        std::cout << "solve_time_s=" << stats.solve_time_s << "\n";
        std::cout << "total_time_s=" << stats.total_time_s << "\n";
        std::cout << "assignment_cost=" << stats.assignment_cost << "\n";
        std::cout << "cost_evaluations=" << stats.cost_evaluations << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
