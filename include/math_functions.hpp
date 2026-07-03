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

#include <array>
#include <cmath>

constexpr int grid_idx(int i, int j, int k, int nx, int ny, int nz) noexcept {
	(void)nx;
	return (i * ny * nz) + (j * nz) + k;
}

template <typename T>
inline T interpolate(T min, T max, T i, T n) noexcept {
	T x = min + (max - min) * i / n;
	return x;
}

template <typename T>
inline T dot(std::array<T, 3> a, std::array<T, 3> b) noexcept {
	T x = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
	return x;
}

template <typename T>
inline T magnitude(std::array<T, 3> a) noexcept {
	T x = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
	return x;
}

template <typename T>
inline std::array<T, 3> cross(std::array<T, 3> a, std::array<T, 3> b) noexcept {
	std::array<T, 3> x = {
		a[1] * b[2] - a[2] * b[1],
		a[2] * b[0] - a[0] * b[2],
		a[0] * b[1] - a[1] * b[0]
	};
	return x;
}

template <typename T>
inline std::array<T, 3> operator+(std::array<T, 3> lhs, const std::array<T, 3> rhs) noexcept {
	std::array<T, 3> x = { lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2] };
	return x;
}

template <typename T>
inline std::array<T, 3> operator-(std::array<T, 3> lhs, const std::array<T, 3> rhs) noexcept {
	std::array<T, 3> x = { lhs[0] - rhs[0], lhs[1] - rhs[1], lhs[2] - rhs[2] };
	return x;
}

template <typename T>
inline std::array<T, 3> &operator+=(std::array<T, 3> &lhs, const std::array<T, 3> &rhs) noexcept {
	lhs[0] += rhs[0]; lhs[1] += rhs[1]; lhs[2] += rhs[2];
	return lhs;
}

template <typename T>
inline std::array<T, 3> operator*(std::array<T, 3> lhs, T b) noexcept {
	std::array<T, 3> x = { lhs[0] *= b, lhs[1] *= b, lhs[2] *= b };
	return x;
}

#endif