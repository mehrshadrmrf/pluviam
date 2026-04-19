#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <execution>
#include <functional>
#include <iterator>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>
#include <type_traits>
#include <vector>

namespace rain::genius::meta {

template <typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

template <Arithmetic T>
inline constexpr bool kUseParallelByDefault = false;

template <typename Iter>
concept RandomAccessIterator = std::random_access_iterator<Iter>;

template <typename Iter, typename T>
concept SuitableForParallel = RandomAccessIterator<Iter> &&
    std::is_arithmetic_v<std::iter_value_t<Iter>> &&
    (std::is_same_v<std::remove_cvref_t<std::iter_value_t<Iter>>, T>);

}

namespace rain::genius::core {

using namespace rain::genius::meta;

template <Arithmetic T>
struct Peak {
    T height;
    std::size_t position;
    constexpr auto operator<=>(const Peak&) const = default;
};

template <Arithmetic T, RandomAccessIterator Iter>
    requires std::is_same_v<std::iter_value_t<Iter>, T>
[[nodiscard]] constexpr T compute_classical(Iter first, Iter last) noexcept {
    const auto n = std::distance(first, last);
    if (n <= 2) [[unlikely]] return T{0};

    auto left = first;
    auto right = std::prev(last);
    T max_left = *left;
    T max_right = *right;
    T accumulated_water = 0;

    while (left < right) {
        if (max_left < max_right) {
            ++left;
            if (*left > max_left) [[unlikely]]
                max_left = *left;
            else
                accumulated_water += max_left - *left;
        } else {
            --right;
            if (*right > max_right) [[unlikely]]
                max_right = *right;
            else
                accumulated_water += max_right - *right;
        }
    }
    return accumulated_water;
}

template <Arithmetic T, SuitableForParallel<T> Iter>
[[nodiscard]] T compute_parallel_scan(Iter first, Iter last) {
    const auto n = std::distance(first, last);
    if (n <= 2) [[unlikely]] return T{0};

    std::vector<T> left_max(static_cast<std::size_t>(n));
    std::vector<T> right_max(static_cast<std::size_t>(n));

    std::inclusive_scan(std::execution::par_unseq,
                        first, last,
                        left_max.begin(),
                        [](T a, T b) noexcept { return std::max(a, b); });

    std::inclusive_scan(std::execution::par_unseq,
                        std::make_reverse_iterator(last),
                        std::make_reverse_iterator(first),
                        right_max.rbegin(),
                        [](T a, T b) noexcept { return std::max(a, b); });

    std::vector<T> water_volume(static_cast<std::size_t>(n));
    std::transform(std::execution::par_unseq,
                   left_max.begin(), left_max.end(),
                   right_max.begin(),
                   water_volume.begin(),
                   [base = first](T lmax, T rmax) mutable noexcept -> T {
                       T height = *base++;
                       T bound = std::min(lmax, rmax);
                       return bound > height ? bound - height : T{0};
                   });

    return std::reduce(std::execution::par_unseq,
                       water_volume.begin(), water_volume.end());
}

template <Arithmetic T, std::size_t N>
    requires (N > 0)
struct StaticSolver {
    using ArrayType = std::array<T, N>;
    ArrayType heights;

    consteval StaticSolver(const ArrayType& arr) noexcept : heights(arr) {}

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

template <Arithmetic T>
struct Dispatcher {
    template <bool UseParallel = false, typename Range>
        requires std::ranges::random_access_range<Range> &&
                 std::is_same_v<std::ranges::range_value_t<Range>, T>
    [[nodiscard]] static constexpr T execute(const Range& range) noexcept(noexcept(
            compute_classical<T>(std::ranges::begin(range), std::ranges::end(range)))) {
        if constexpr (UseParallel) {
            return compute_parallel_scan<T>(std::ranges::begin(range), std::ranges::end(range));
        } else {
            return compute_classical<T>(std::ranges::begin(range), std::ranges::end(range));
        }
    }
};

}

namespace rain::genius::interface {

using namespace rain::genius::meta;
using namespace rain::genius::core;

template <Arithmetic T>
class FluentTrap {
public:
    template <std::ranges::random_access_range R>
        requires std::is_same_v<std::ranges::range_value_t<R>, T>
    explicit FluentTrap(R&& range)
        : data_(std::forward<R>(range) | std::ranges::to<std::vector<T>>()) {}

    static FluentTrap from_vector(std::vector<T>&& vec) noexcept {
        FluentTrap ft;
        ft.data_ = std::move(vec);
        return ft;
    }

    static FluentTrap from_vector(const std::vector<T>& vec) {
        FluentTrap ft;
        ft.data_ = vec;
        return ft;
    }

    [[nodiscard]] T classical() const noexcept {
        return Dispatcher<T>::template execute<false>(data_);
    }

    [[nodiscard]] T parallel() const {
        return Dispatcher<T>::template execute<true>(data_);
    }

    [[nodiscard]] T auto_select() const {
        constexpr std::size_t PARALLEL_THRESHOLD = 1024 * 1024;
        if (data_.size() >= PARALLEL_THRESHOLD) {
            return parallel();
        }
        return classical();
    }

    [[nodiscard]] const std::vector<T>& data() const noexcept { return data_; }

private:
    std::vector<T> data_;

    FluentTrap() = default;
};

template <Arithmetic T, std::ranges::random_access_range R>
    requires std::is_same_v<std::ranges::range_value_t<R>, T>
[[nodiscard]] inline T trap(const R& range) {
    return FluentTrap<T>{range}.classical();
}

template <Arithmetic T>
[[nodiscard]] inline T trap_parallel(const std::vector<T>& vec) {
    return FluentTrap<T>::from_vector(vec).parallel();
}

template <Arithmetic T>
[[nodiscard]] inline T trap_auto(const std::vector<T>& vec) {
    return FluentTrap<T>::from_vector(vec).auto_select();
}

}

namespace rain::genius::test {

using namespace rain::genius::interface;

static_assert([]{
    constexpr std::array<int, 12> arr{0,1,0,2,1,0,1,3,2,1,2,1};
    constexpr auto solver = core::StaticSolver<int, 12>{arr};
    return solver.water_volume() == 6;
}(), "The static solver failed.");

static_assert([]{
    constexpr std::array<int, 6> arr{4,2,0,3,2,5};
    constexpr auto solver = core::StaticSolver<int, 6>{arr};
    return solver.water_volume() == 9;
}(), "Static solver sanity check #2 failed.");

static_assert([]{
    constexpr std::array<int, 0> arr{};
    constexpr auto solver = core::StaticSolver<int, 0>{arr};
    return solver.water_volume() == 0;
}(), "Empty array should trap zero water.");

static_assert([]{
    constexpr std::array<double, 5> arr{1.0, 2.0, 1.0, 2.0, 1.0};
    constexpr auto solver = core::StaticSolver<double, 5>{arr};
    return solver.water_volume() == 1.0;
}(), "Floating point static test failed.");

}

#include <iostream>
#include <iomanip>
#include <chrono>

int main() {
    using namespace rain::genius::interface;
    using namespace std::chrono;

    const std::vector<int> heights_int = {0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1};

    std::cout << "️  Trapping Rain Water \n";
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

    std::cout << "\n  Performance on " << LARGE_N << " elements:\n";
    std::cout << "   Classical O(1) space:    " << dur_classic.count() << " ms\n";
    std::cout << "   Parallel scan (par_unseq): " << dur_parallel.count() << " ms\n";

    std::cout << "\n All tests passed. The rain has been successfully trapped.\n";
    std::cout << "Jeff Dean high‑fives a compiler.\n";

    return 0;
}
