// Derived from code by jatinchowdhury18 (2020)
// Licensed under the BSD 3-Clause License
// https://github.com/jatinchowdhury18/RTNeural
#pragma once
#include "xsimd/xsimd.hpp"
#include <vector>

template <typename T>
constexpr T ceil_div(T x, T y)
{
    return (x + y - 1) / y;
}

template <typename T, std::size_t Alignment>
void set_values(
    const std::vector<T>& weights,
    xsimd::simd_type<T>* simd_weights,  // pointer to SIMD weight array
    int total_size,                     // total number of scalar weights (e.g. 16)
    int batch_count                     // number of SIMD batches (e.g. 4 -> 16 // 4)
)
{
    using v_type = xsimd::simd_type<T>;
    constexpr std::size_t alignment = Alignment;
    constexpr int v_size = static_cast<int>(v_type::size);

    for (int batch_idx = 0; batch_idx < batch_count; ++batch_idx) {
        alignas(alignment) T tmp[v_size] = { 0 };  // temporary buffer for one SIMD batch

        for (int lane = 0; lane < v_size; ++lane) {
            int idx = batch_idx * v_size + lane;
            if (idx < total_size) {
                tmp[lane] = static_cast<T>(weights[idx]);
            }
            else {
                tmp[lane] = T(0);  // zero padding for leftover lanes
            }
        }

        simd_weights[batch_idx] = xsimd::load_aligned(tmp);
    }
}