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

#include <array>
#include <complex>

#include "physics.hpp"
#include "laguerre_gauss.hpp"
#include "math_functions.hpp"

template <typename T>
void test_u(ScalarField<T> &field, const Laser<T> &laser) {
	int nx = field.num[0], ny = field.num[1], nz = field.num[2];
	T r_max_x = field.r_max[0], r_max_y = field.r_max[1], r_max_z = field.r_max[2];
	#pragma omp parallel for collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::array<T, 3> r_i = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				std::complex<T> u_i = compute_u(laser, r_i);
				int idx = grid_idx(i, j, k, nx, ny, nz);
				field.v[idx] = real(u_i);
			}
		}
	}
}

template <typename T>
void compute_e_field(VectorField<T> &field, const Laser<T> &laser, T t) {
	int nx = field.num[0], ny = field.num[1], nz = field.num[2];
	T r_max_x = field.r_max[0], r_max_y = field.r_max[1], r_max_z = field.r_max[2];
	#pragma omp parallel for collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::array<T, 3> r_i = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				
				std::array<T, 3> e_vec = compute_e(laser, r_i, t);
				
				int idx = grid_idx(i, j, k, nx, ny, nz);
				field.x[idx] = e_vec[0];
				field.y[idx] = e_vec[1];
				field.z[idx] = e_vec[2];
			}
		}
	}
}

template void test_u<double>(ScalarField<double> &field, const Laser<double> &laser);
template void compute_e_field<double>(VectorField<double> &field, const Laser<double> &laser, double t);
 
template void test_u<float>(ScalarField<float> &field, const Laser<float> &laser);
template void compute_e_field<float>(VectorField<float> &field, const Laser<float> &laser, float t);