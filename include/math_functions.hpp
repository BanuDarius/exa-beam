/* MIT License

Copyright (c) 2026 Banu Darius-Matei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#ifndef MATH_FUNCTIONS_H
#define MATH_FUNCTIONS_H

#include <cmath>
#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>

__device__ __host__ constexpr std::size_t grid_idx(int i, int j, int k, int nx, int ny, int nz) noexcept {
	(void)nx;
	return (static_cast<std::size_t>(i) * ny * nz) + (static_cast<std::size_t>(j) * nz) + k;
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
	using std::sqrt;
	T x = sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
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
__device__ __host__ inline cuda::std::array<T, 3> operator+(cuda::std::array<T, 3> a, const cuda::std::array<T, 3> b) noexcept {
	cuda::std::array<T, 3> x = { a[0] + b[0], a[1] + b[1], a[2] + b[2] };
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> operator-(cuda::std::array<T, 3> a, const cuda::std::array<T, 3> b) noexcept {
	cuda::std::array<T, 3> x = { a[0] - b[0], a[1] - b[1], a[2] - b[2] };
	return x;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> &operator+=(cuda::std::array<T, 3> &a, const cuda::std::array<T, 3> &b) noexcept {
	a[0] += b[0]; a[1] += b[1]; a[2] += b[2];
	return a;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> operator*(cuda::std::array<T, 3> a, T b) noexcept {
	cuda::std::array<T, 3> x = { a[0] * b, a[1] * b, a[2] * b };
	return x;
}

#endif