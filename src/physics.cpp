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

template <std::floating_point T>
void compute_lz(Particles<T> &particles, ScalarField<T> &lz_field) {
	int nx = particles.num[0], ny = particles.num[1], nz = particles.num[2];
	#pragma omp parallel for collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				int idx = grid_idx(i, j, k, nx, ny, nz);
				std::array<T, 3> r_vec = particles.get_cpu_view().get_position(idx);
				std::array<T, 3> u_vec = particles.get_cpu_view().get_velocity(idx);
				T x = r_vec[0], y = r_vec[1];
				T ux = u_vec[0], uy = u_vec[1];
				
				T lz = m_e<T> * (x * uy - y * ux);
				
				lz_field.get_cpu_view().set_field(lz, idx);
			}
		}
	}
}

template <std::floating_point T>
void compute_u_field(ComplexScalarField<T> &u_field, const Laser<T> &laser) {
	int nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	T r_max_x = u_field.r_max[0], r_max_y = u_field.r_max[1], r_max_z = u_field.r_max[2];
	#pragma omp parallel for collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::array<T, 3> r_vec = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				T z = r_vec[2], z_r = laser.z_r, w0 = laser.w0;
				T r_z = compute_r_z(z, z_r);
				T w_z = compute_w_z(w0, z, z_r);
				
				std::complex<T> u_i = compute_u(laser, r_vec, r_z, w_z);
				int idx = grid_idx(i, j, k, nx, ny, nz);
				
				u_field.get_cpu_view().set_field(u_i, idx);
			}
		}
	}
}

template <std::floating_point T>
void compute_eb_field(VectorField<T> &e_field, VectorField<T> &b_field, const ComplexScalarField<T> &u_field, const Laser<T> &laser, T t) {
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	T r_max_x = e_field.r_max[0], r_max_y = e_field.r_max[1], r_max_z = e_field.r_max[2];
	#pragma omp parallel for collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::array<T, 3> r_vec = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				int idx = grid_idx(i, j, k, nx, ny, nz);
				
				EBVectors<T> eb_vec = compute_eb_field(u_field, laser, r_vec, t, idx);
				std::array<T, 3> e_vec = eb_vec.e;
				std::array<T, 3> b_vec = eb_vec.b;
				
				e_field.get_cpu_view().set_field(e_vec, idx);
				b_field.get_cpu_view().set_field(b_vec, idx);
			}
		}
	}
}

template void compute_lz<double>(Particles<double> &particles, ScalarField<double> &lz_field);
template void compute_u_field<double>(ComplexScalarField<double> &u_field, const Laser<double> &laser);
template void compute_eb_field<double>(VectorField<double> &e_field, VectorField<double> &b_field, const ComplexScalarField<double> &u_field, const Laser<double> &laser, double t);
 
template void compute_lz<float>(Particles<float> &particles, ScalarField<float> &lz_field);
template void compute_u_field<float>(ComplexScalarField<float> &u_field, const Laser<float> &laser);
template void compute_eb_field<float>(VectorField<float> &e_field, VectorField<float> &b_field,  const ComplexScalarField<float> &u_field, const Laser<float> &laser, float t);