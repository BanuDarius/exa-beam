// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/cmath>

__device__ __host__ constexpr std::size_t grid_idx(int i, int j, int k, int nx, int ny, int nz) noexcept {
	(void)nz;
	return (static_cast<std::size_t>(k) * nx * ny) + (static_cast<std::size_t>(j) * nx) + i;
}

template <std::floating_point T>
__device__ __host__ inline T interpolate(T min, T max, T i, T n) noexcept {
	T x = min + (max - min) * i / n;
	return x;
}

template <std::floating_point T>
__device__ __host__ inline T dot(cuda::std::array<T, 3> a, cuda::std::array<T, 3> b) noexcept {
	T x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	return x;
}

template <std::floating_point T>
__device__ __host__ inline T magnitude(cuda::std::array<T, 3> a) noexcept {
	T x = cuda::std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> cross(cuda::std::array<T, 3> a, cuda::std::array<T, 3> b) noexcept {
	cuda::std::array<T, 3> x = {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> operator+(cuda::std::array<T, 3> a, cuda::std::array<T, 3> b) noexcept {
	cuda::std::array<T, 3> x = { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> operator-(cuda::std::array<T, 3> a, cuda::std::array<T, 3> b) noexcept {
	cuda::std::array<T, 3> x = { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> &operator+=(cuda::std::array<T, 3> &a, const cuda::std::array<T, 3> &b) noexcept {
	a[0] += b[0]; a[1] += b[1]; a[2] += b[2];
	return a;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> &operator-=(cuda::std::array<T, 3> &a, const cuda::std::array<T, 3> &b) noexcept {
	a[0] -= b[0]; a[1] -= b[1]; a[2] -= b[2];
	return a;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> operator*(cuda::std::array<T, 3> a, T b) noexcept {
	cuda::std::array<T, 3> x = { a[0] * b, a[1] * b, a[2] * b };
	return x;
}

#endif