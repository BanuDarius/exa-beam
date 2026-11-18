// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef GATHER_H
#define GATHER_H

#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/cmath>
#include <cuda/std/algorithm>

#include "structs.hpp"

template <std::floating_point T> struct WeightsCIC { int i0; T w0, w1; };

template <std::floating_point T>
__device__ __host__ inline WeightsCIC<T> cic_weights(T xi, int n) noexcept {
    int idx_base = static_cast<int>(cuda::std::floor(xi));
    int i0 = cuda::std::clamp(idx_base, 0, n - 2);
    T delta = cuda::std::clamp(xi - static_cast<T>(i0), T(0.0), T(1.0));
    return { i0, T(1.0) - delta, delta };
}

template <int Axis, std::floating_point T>
__device__ __host__ inline T get_component(const VectorFieldView<T> &e, int i, int j, int k) noexcept {
    if constexpr(Axis == 0) { return e.get_x(i, j, k); }
    else if constexpr(Axis == 1) { return e.get_y(i, j, k); }
    else { return e.get_z(i, j, k); }
}

template <int Axis, std::floating_point T>
__device__ __host__ inline T cic_interpolate(const VectorFieldView<T> &e, WeightsCIC<T> wx, WeightsCIC<T> wy, WeightsCIC<T> wz) noexcept {
    int i0 = wx.i0, i1 = i0 + 1;
    int j0 = wy.i0, j1 = j0 + 1;
    int k0 = wz.i0, k1 = k0 + 1;
    
    T c000 = get_component<Axis>(e, i0, j0, k0);
    T c001 = get_component<Axis>(e, i0, j0, k1);
    T c010 = get_component<Axis>(e, i0, j1, k0);
    T c011 = get_component<Axis>(e, i0, j1, k1);
    T c100 = get_component<Axis>(e, i1, j0, k0);
    T c101 = get_component<Axis>(e, i1, j0, k1);
    T c110 = get_component<Axis>(e, i1, j1, k0);
    T c111 = get_component<Axis>(e, i1, j1, k1);
    
    T c00 = wx.w0 * c000 + wx.w1 * c100;
    T c01 = wx.w0 * c001 + wx.w1 * c101;
    T c10 = wx.w0 * c010 + wx.w1 * c110;
    T c11 = wx.w0 * c011 + wx.w1 * c111;
    
    T c0 = wy.w0 * c00 + wy.w1 * c10;
    T c1 = wy.w0 * c01 + wy.w1 * c11;
    
    return wz.w0 * c0 + wz.w1 * c1;
}

template <std::floating_point T>
__device__ __host__ inline EBVector3<T> cic_gather(Vector3<T> r_vec, const VectorFieldView<T> &e, const VectorFieldView<T> &b) noexcept {
    if(cuda::std::abs(r_vec[0]) > e.r_max[0] || cuda::std::abs(r_vec[1]) > e.r_max[1] || cuda::std::abs(r_vec[2]) > e.r_max[2]) {
        return { EBVector3<T>{} };
    }
    int nx = e.num[0], ny = e.num[1], nz = e.num[2];
    Vector3<T> g = {
        (r_vec[0] + e.r_max[0]) * e.inv_dx[0],
        (r_vec[1] + e.r_max[1]) * e.inv_dx[1],
        (r_vec[2] + e.r_max[2]) * e.inv_dx[2]
    };
    
    WeightsCIC wx_0 = cic_weights(g[0], nx);
    WeightsCIC wy_0 = cic_weights(g[1], ny);
    WeightsCIC wz_0 = cic_weights(g[2], nz);
    
    WeightsCIC wx_h = cic_weights(g[0] - T(0.5), nx - 1);
    WeightsCIC wy_h = cic_weights(g[1] - T(0.5), ny - 1);
    WeightsCIC wz_h = cic_weights(g[2] - T(0.5), nz - 1);
    
    Vector3<T> e_vec = {
        cic_interpolate<0>(e, wx_h, wy_0, wz_0),
        cic_interpolate<1>(e, wx_0, wy_h, wz_0),
        cic_interpolate<2>(e, wx_0, wy_0, wz_h)
    }, b_vec = {
        cic_interpolate<0>(b, wx_0, wy_h, wz_h),
        cic_interpolate<1>(b, wx_h, wy_0, wz_h),
        cic_interpolate<2>(b, wx_h, wy_h, wz_0)
    };
    return { e_vec, b_vec };
}

#endif
