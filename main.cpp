#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <vector>
#include <omp.h>
#include <iostream>
#include <iomanip>
#include <chrono>


// rain::genius::meta - Compile‑time type constraints and concepts

namespace rain::genius::meta {

/**
 * @concept Arithmetic
 * @brief Satisfied by any type for which std::is_arithmetic_v is true.
 *        Used to restrict templates to integral or floating‑point types.
 * @tparam T the type to check
 */
template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

/**
 * @concept RandomAccessIterator
 * @brief Satisfied by any type that models std::random_access_iterator.
 * @tparam Iter the iterator type to check
 */
template <typename Iter>
concept RandomAccessIterator = std::random_access_iterator<Iter>;

/**
 * @concept SuitableForParallel
 * @brief Combines requirements for parallel algorithms:
 *        - the iterator is random‑access
 *        - its value type is arithmetic
 *        - exactly matches the explicitly given type T.
 * @tparam Iter iterator type
 * @tparam T    element type, must match iterator value type
 */
template <typename Iter, typename T>
concept SuitableForParallel = RandomAccessIterator<Iter> &&
    std::is_arithmetic_v<std::iter_value_t<Iter>> &&
    std::is_same_v<std::iter_value_t<Iter>, T>;

} // namespace rain::genius::meta


// rain::genius::core - Algorithm implementations

namespace rain::genius::core {

using namespace rain::genius::meta;

/**
 * @brief Classic two‑pointer rain water trapping.
 *        O(n) time, O(1) additional space.
 *
 * @tparam T   arithmetic type of the heights.
 * @tparam Iter random‑access iterator whose value type is T.
 * @param first, last range of height values (at least 2 elements, else returns 0).
 * @return total trapped water.
 */
template <Arithmetic T, RandomAccessIterator Iter>
    requires std::is_same_v<std::iter_value_t<Iter>, T>
[[nodiscard]] constexpr T compute_classical(Iter first, Iter last) noexcept {
    const auto n = std::distance(first, last);
    if (n <= 2) [[unlikely]] return T{0};

    auto left = first;
    auto right = std::prev(last);
    T max_left = *left;
    T max_right = *right;
    T water = 0;

    while (left < right) {
        if (max_left < max_right) {
            ++left;
            if (*left > max_left) [[unlikely]]
                max_left = *left;
            else
                water += max_left - *left;
        } else {
            --right;
            if (*right > max_right) [[unlikely]]
                max_right = *right;
            else
                water += max_right - *right;
        }
    }
    return water;
}


// ---------------------------------------------------------------------
//  Parallel prefix max (inclusive scan) helper
// ---------------------------------------------------------------------
namespace detail {

    /**
     * @brief Computes inclusive prefix maximum of the array in parallel
     *        using OpenMP.  Writes the result into out[].
     *
     * @tparam T  arithmetic element type.
     * @param first  pointer to contiguous input data (heap‑allocated vector).
     * @param n      number of elements.
     * @param out    output array of size n, already allocated.
     *
     * @note The implementation assumes that out is pre‑allocated with at least n elements.
     *       It uses a chunk‑based local scan followed by a global fix‑up.
     */
    template <Arithmetic T>
    void parallel_inclusive_scan_max(const T* __restrict first, std::size_t n,
                                     T* __restrict out) {
        int actual_nt = 1;
        std::vector<T> chunk_max(omp_get_max_threads());

        #pragma omp parallel
        {
            #pragma omp single
            actual_nt = omp_get_num_threads();

            const int tid       = omp_get_thread_num();
            const std::size_t chunk = (n + actual_nt - 1) / actual_nt;
            const std::size_t start  = tid * chunk;
            const std::size_t end    = std::min(start + chunk, n);

            if (start < end) {
                out[start] = first[start];
                for (std::size_t i = start + 1; i < end; ++i)
                    out[i] = std::max(out[i - 1], first[i]);

                chunk_max[tid] = out[end - 1];
            }
        }

        // Propagate maximums across chunk boundaries.
        for (int i = 1; i < actual_nt; ++i)
            if (chunk_max[i] < chunk_max[i - 1])
                chunk_max[i] = chunk_max[i - 1];

        #pragma omp parallel
        {
            const int tid       = omp_get_thread_num();
            const std::size_t chunk = (n + actual_nt - 1) / actual_nt;
            const std::size_t start  = tid * chunk;
            const std::size_t end    = std::min(start + chunk, n);

            if (tid > 0 && start < end) {
                const T offset = chunk_max[tid - 1];
                for (std::size_t i = start; i < end; ++i)
                    if (out[i] < offset) out[i] = offset;
            }
        }
    }

} // namespace detail


// ---------------------------------------------------------------------
//  Parallel trapping using prefix‑max arrays
// ---------------------------------------------------------------------
/**
 * @brief Parallel rain water trapping using prefix / suffix maximum scans.
 *        O(n) work, O(n) additional memory.
 *
 * @tparam T   arithmetic type.
 * @tparam Iter random‑access iterator satisfying SuitableForParallel<T>.
 * @param first, last the height range.
 * @return total trapped water.
 */
template <Arithmetic T, SuitableForParallel<T> Iter>
[[nodiscard]] T compute_parallel_scan(Iter first, Iter last) {
    const auto n = static_cast<std::size_t>(std::distance(first, last));
    if (n <= 2) [[unlikely]] return T{0};

    // For tiny inputs the parallel overhead dominates – fall back to classical.
    constexpr std::size_t MIN_PARALLEL_SIZE = 4096;
    if (n < MIN_PARALLEL_SIZE) {
        return compute_classical<T>(first, last);
    }

    // 1) left‑to‑right prefix maximum
    std::vector<T> left_max(n);
    detail::parallel_inclusive_scan_max(&(*first), n, left_max.data());

    // 2) right‑to‑left prefix maximum (computed by reversing the array)
    std::vector<T> reversed(n);
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        reversed[i] = first[n - 1 - i];

    std::vector<T> rev_prefix(n);
    detail::parallel_inclusive_scan_max(reversed.data(), n, rev_prefix.data());

    std::vector<T> right_max(n);
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i)
        right_max[i] = rev_prefix[n - 1 - i];

    // 3) Water above each bar = min(left_max, right_max) - height
    std::vector<T> water_volume(n);
    #pragma omp parallel for
    for (std::size_t i = 0; i < n; ++i) {
        T bound = std::min(left_max[i], right_max[i]);
        water_volume[i] = (bound > first[i]) ? (bound - first[i]) : T{0};
    }

    // 4) Summation with reduction
    T total = 0;
    #pragma omp parallel for reduction(+:total)
    for (std::size_t i = 0; i < n; ++i)
        total += water_volume[i];

    return total;
}


// ---------------------------------------------------------------------
//  Compile‑time (constexpr) solver over std::array
// ---------------------------------------------------------------------
/**
 * @brief Compile‑time rain water trapping for a fixed‑size array.
 *        Evaluated entirely at compile time (consteval) when used with
 *        constant expressions.  No runtime cost.
 *
 * @tparam T  arithmetic type.
 * @tparam N  number of elements (must be > 0).
 */
template <Arithmetic T, std::size_t N>
    requires (N > 0)
struct StaticSolver {
    using ArrayType = std::array<T, N>;
    ArrayType heights;

    /**
     * @param arr the height array (typically a constexpr array).
     */
    consteval StaticSolver(const ArrayType& arr) noexcept : heights(arr) {}

    /**
     * @return trapped water volume computed at compile time.
     */
    [[nodiscard]] consteval T water_volume() const noexcept {
        if constexpr (N <= 2) return T{0};

        T max_left = heights[0];
        T max_right = heights[N - 1];
        std::size_t l = 0, r = N - 1;
        T water = 0;

        while (l < r) {
            if (max_left < max_right) {
                ++l;
                if (heights[l] > max_left)
                    max_left = heights[l];
                else
                    water += max_left - heights[l];
            } else {
                --r;
                if (heights[r] > max_right)
                    max_right = heights[r];
                else
                    water += max_right - heights[r];
            }
        }
        return water;
    }
};


// ---------------------------------------------------------------------
//  Dispatcher: compile‑time switch between classical and parallel
// ---------------------------------------------------------------------
/**
 * @brief Compile‑time dispatch helper.
 *        Allows selecting between classical and parallel algorithms
 *        through a template boolean parameter.
 *
 * @tparam T  arithmetic element type.
 */
template <Arithmetic T>
struct Dispatcher {
    /**
     * @brief Execute the chosen algorithm on a range.
     *
     * @tparam UseParallel  if true, uses the parallel scan version.
     * @tparam Range        a random‑access range with value_type T.
     * @param range  input range of heights.
     * @return trapped water.
     */
    template <bool UseParallel = false, typename Range>
        requires std::ranges::random_access_range<Range> &&
                 std::is_same_v<std::ranges::range_value_t<Range>, T>
    [[nodiscard]] static T execute(const Range& range) {
        if constexpr (UseParallel) {
            return compute_parallel_scan<T>(std::ranges::begin(range), std::ranges::end(range));
        } else {
            return compute_classical<T>(std::ranges::begin(range), std::ranges::end(range));
        }
    }
};

} // namespace rain::genius::core


// rain::genius::interface - Fluent API and free functions

namespace rain::genius::interface {

using namespace rain::genius::meta;
using namespace rain::genius::core;

/**
 * @brief Fluent interface for rain water trapping.
 *        Holds a copy of the height vector and offers classical(),
 *        parallel(), and auto_select() methods.
 *
 * @tparam T arithmetic type of the heights.
 */
template <Arithmetic T>
class FluentTrap {
public:
    /**
     * @brief Construct from a random‑access range.
     * @tparam R range type.
     * @param range input range of heights.
     */
    template <std::ranges::random_access_range R>
        requires std::is_same_v<std::ranges::range_value_t<R>, T>
    explicit FluentTrap(R&& range)
        : data_(std::ranges::begin(range), std::ranges::end(range)) {}

    /**
     * @brief Create a FluentTrap by moving an existing vector.
     * @param vec rvalue reference to a vector.
     * @return FluentTrap owning the data.
     */
    static FluentTrap from_vector(std::vector<T>&& vec) noexcept {
        FluentTrap ft;
        ft.data_ = std::move(vec);
        return ft;
    }

    /**
     * @brief Create a FluentTrap by copying an existing vector.
     * @param vec const reference to a vector.
     * @return FluentTrap with a copy of the data.
     */
    static FluentTrap from_vector(const std::vector<T>& vec) {
        FluentTrap ft;
        ft.data_ = vec;
        return ft;
    }

    /**
     * @return trapped water using the classical O(1)‑space algorithm.
     */
    [[nodiscard]] T classical() const noexcept {
        return Dispatcher<T>::template execute<false>(data_);
    }

    /**
     * @return trapped water using the parallel scan algorithm.
     */
    [[nodiscard]] T parallel() const {
        return Dispatcher<T>::template execute<true>(data_);
    }

    /**
     * @brief Automatically select classical or parallel based on a size threshold.
     *        For sizes >= 1'048'576 the parallel version is used.
     * @return trapped water.
     */
    [[nodiscard]] T auto_select() const {
        constexpr std::size_t PARALLEL_THRESHOLD = 1024 * 1024;
        if (data_.size() >= PARALLEL_THRESHOLD) {
            return parallel();
        }
        return classical();
    }

    /**
     * @return const reference to the stored height vector.
     */
    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }

private:
    std::vector<T> data_;
    FluentTrap() = default;
};


// ---------------------------------------------------------------------
//  Convenience free functions
// ---------------------------------------------------------------------

/**
 * @brief Classical rain water trap on a range (free function).
 * @tparam R range type with arithmetic value_type.
 * @param range input range of heights.
 * @return trapped water.
 */
template <std::ranges::random_access_range R>
    requires Arithmetic<std::ranges::range_value_t<R>>
[[nodiscard]] inline auto trap(const R& range) {
    using T = std::ranges::range_value_t<R>;
    return FluentTrap<T>{range}.classical();
}

/**
 * @brief Parallel trap on a vector (free function).
 * @tparam T arithmetic type.
 * @param vec vector of heights.
 * @return trapped water.
 */
template <Arithmetic T>
[[nodiscard]] inline T trap_parallel(const std::vector<T>& vec) {
    return FluentTrap<T>::from_vector(vec).parallel();
}

/**
 * @brief Auto‑select trap on a vector (free function).
 * @tparam T arithmetic type.
 * @param vec vector of heights.
 * @return trapped water.
 */
template <Arithmetic T>
[[nodiscard]] inline T trap_auto(const std::vector<T>& vec) {
    return FluentTrap<T>::from_vector(vec).auto_select();
}

} // namespace rain::genius::interface



// rain::genius::test - Compile‑time self‑tests
namespace rain::genius::test {

using namespace rain::genius::interface;

static_assert([]{
    constexpr std::array<int, 12> arr{0,1,0,2,1,0,1,3,2,1,2,1};
    constexpr auto solver = core::StaticSolver<int, 12>{arr};
    return solver.water_volume() == 6;
}(), "Static solver failed.");

static_assert([]{
    constexpr std::array<int, 6> arr{4,2,0,3,2,5};
    constexpr auto solver = core::StaticSolver<int, 6>{arr};
    return solver.water_volume() == 9;
}(), "Static solver sanity check #2 failed.");

static_assert([]{
    constexpr std::array<double, 5> arr{1.0, 2.0, 1.0, 2.0, 1.0};
    constexpr auto solver = core::StaticSolver<double, 5>{arr};
    return solver.water_volume() == 1.0;
}(), "Floating point static test failed.");

} // namespace rain::genius::test



// Example / benchmark

int main() {
    using namespace rain::genius::interface;
    using namespace std::chrono;

    const std::vector<int> heights_int = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};

    std::cout << "  Trapping Rain Water \n";
    std::cout << "==============================================\n\n";

    std::cout << "1. Free function trap():              " << trap(heights_int) << '\n';
    std::cout << "2. Fluent interface classical():      "
              << FluentTrap<int>::from_vector(heights_int).classical() << '\n';
    std::cout << "3. Fluent interface parallel():       "
              << FluentTrap<int>::from_vector(heights_int).parallel() << '\n';
    std::cout << "4. Fluent auto_select():              "
              << FluentTrap<int>::from_vector(heights_int).auto_select() << '\n';

    constexpr std::array<int, 12> arr_int = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};
    constexpr auto static_solver = rain::genius::core::StaticSolver<int, 12>{arr_int};
    constexpr int static_result = static_solver.water_volume();
    std::cout << "5. Compile‑time static evaluation:    " << static_result << " (zero runtime cost)\n";

    const std::vector<double> heights_dbl = {0.0, 1.0, 0.0, 2.0, 1.0, 0.0, 1.0, 3.0, 2.0, 1.0, 2.0, 1.0};
    std::cout << "6. Double precision parallel:         "
              << std::setprecision(1) << std::fixed
              << trap_parallel(heights_dbl) << '\n';

    constexpr std::size_t LARGE_N = 10'000'000;
    std::vector<int> large_vec(LARGE_N);
    std::generate(large_vec.begin(), large_vec.end(), [n = 0]() mutable {
        return (n++ % 100) * (n % 7) / 13;
    });

    auto start = high_resolution_clock::now();
    [[maybe_unused]] volatile int result_classic = FluentTrap<int>::from_vector(large_vec).classical();
    auto end = high_resolution_clock::now();
    auto dur_classic = duration_cast<milliseconds>(end - start);

    start = high_resolution_clock::now();
    [[maybe_unused]] volatile int result_parallel = FluentTrap<int>::from_vector(large_vec).parallel();
    end = high_resolution_clock::now();
    auto dur_parallel = duration_cast<milliseconds>(end - start);

    std::cout << "\n Performance on " << LARGE_N << " elements:\n";
    std::cout << "   Classical O(1) space:    " << dur_classic.count() << " ms\n";
    std::cout << "   Parallel scan (OpenMP):  " << dur_parallel.count() << " ms\n";

    std::cout << "\n All tests passed. The rain has been successfully trapped.\n";
    return 0;
}
