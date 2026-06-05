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

#ifndef SIM_STRUCTS_H
#define SIM_STRUCTS_H

#include <memory>

#include "math_functions.hpp"

constexpr int string_size = 128;

template <typename T> constexpr T c = T(137.036);
template <typename T> constexpr T pi = T(3.14159265359);

constexpr int grid_idx(int i, int j, int k, int nx, int ny, int nz) noexcept {
	(void)nx;
	return (i * ny * nz) + (j * nz) + k;
}

template <typename T>
struct Particles {
	std::array<int, 3> num;
	std::array<T, 2> r_max;
	std::unique_ptr<T[]> x, y, z, px, py, pz;
	Particles(int nx, int ny, int nz, T r_max_n) {
		r_max = { r_max_n, r_max_n };
		num = { nx, ny, nz };
		std::size_t total = nx * ny * nz;
		x = std::unique_ptr<T[]>(new T[total]);
		y = std::unique_ptr<T[]>(new T[total]);
		z = std::unique_ptr<T[]>(new T[total]);
		px = std::unique_ptr<T[]>(new T[total]);
		py = std::unique_ptr<T[]>(new T[total]);
		pz = std::unique_ptr<T[]>(new T[total]);
		#pragma omp parallel for collapse(3) schedule(static)
		for(int i = 0; i < nx; i++) {
			for(int j = 0; j < ny; j++) {
				for(int k = 0; k < nz; k++) {
					std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
					x[idx] = interpolate(-r_max[0], r_max[0], static_cast<T>(i), static_cast<T>(nx));
					y[idx] = interpolate(-r_max[1], r_max[1], static_cast<T>(j), static_cast<T>(ny));
					z[idx] = T(0.0);
					px[idx] = T(0.0); py[idx] = T(0.0); pz[idx] = T(0.0);
				}
			}
		}
	}
};

template <typename T>
struct Laser {
	int p, m;
	T omega, w0, k, lambda, z_r;
	Laser(int p_n, int m_n, T omega_n, T w0_multiplier) : p(p_n), m(m_n), omega(omega_n) {
		k = omega / c<T>;
		lambda = (T(2.0) * pi<T> * c<T>) / omega;
		w0 = lambda * w0_multiplier;
		z_r = pi<T> * w0 * w0 / lambda;
	}
};

#endif