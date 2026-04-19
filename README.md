# 🌧️ Impluvium · C++23 Metaprogramming & Parallel Scan Showcase

[![Standard](https://img.shields.io/badge/C%2B%2B-23-blue.svg)](https://en.cppreference.com/w/cpp/23)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-cmake%20--build%20--config%20Release-brightgreen)]()
[![Concepts](https://img.shields.io/badge/Concepts-Enabled-ff69b4)]()

> *"Not just trapping rain - trapping complexity within zero-cost abstractions."*

**Impluvium** is a love letter to Modern C++ (C++20/23) disguised as a classic algorithm problem.  
It solves **Trapping Rain Water** with three different mindsets:
1. **Classical O(1) space** - Two pointers, branch prediction hints.
2. **Parallel Heterogeneous** - `std::execution::par_unseq` + `inclusive_scan`.
3. **Compile-time Oracle** - `consteval` static solver for array literals.

But the water is just an excuse. The real project is the **meta-architecture**:
- **`rain::genius::meta`**: Custom `concepts` for arithmetic & iterator safety.
- **`rain::genius::core`**: Zero-overhead dispatching between sequential and parallel backends.
- **`rain::genius::interface`**: Fluent API design (`FluentTrap`) with automatic execution policy selection.

---

## 🧠 Why This Matters (The "Hidden" Part)

This repository demonstrates that **C++ is not just a language; it's a toolkit for building your own abstractions over silicon**.

| Feature | Pedagogical / Professional Signal |
|---------|----------------------------------|
| `consteval StaticSolver` | **Proof-carrying code**: The result is guaranteed before `main()` runs. |
| `std::execution::par_unseq` | **Heterogeneous computing readiness**: Prepared for SIMD + multi-core. |
| `SuitableForParallel` Concept | **Type-driven design**: Prevents accidental misuse of parallel algorithms. |
| `namespace rain::genius` | A nod to the joy of *naming things* in computer science. |

---

## 📦 Quick Start

```bash
git clone https://github.com/yourusername/impluvium.git
cd impluvium
cmake -B build -DCMAKE_CXX_STANDARD=23 -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/impluvium_demo
