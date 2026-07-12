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

#include <cuda/std/array>
#include <cuda/std/complex>

#include "physics.hpp"
#include "laguerre_gauss.hpp"
#include "math_functions.hpp"

template <std::floating_point T>
void compute_lz(ScalarField<T> &lz_field, Particles<T> &particles) noexcept {
	int nx = particles.num[0], ny = particles.num[1], nz = particles.num[2];
	ScalarFieldView lz_field_view = lz_field.get_cpu_view();
	ParticlesView particles_view = particles.get_cpu_view();
	T *__restrict__ ptr_v = std::assume_aligned<mem_align>(lz_field_view.v);
	T *__restrict__ ptr_x = std::assume_aligned<mem_align>(particles_view.x);
	T *__restrict__ ptr_y = std::assume_aligned<mem_align>(particles_view.y);
	T *__restrict__ ptr_z = std::assume_aligned<mem_align>(particles_view.z);
	T *__restrict__ ptr_ux = std::assume_aligned<mem_align>(particles_view.ux);
	T *__restrict__ ptr_uy = std::assume_aligned<mem_align>(particles_view.uy);
	T *__restrict__ ptr_uz = std::assume_aligned<mem_align>(particles_view.uz);
	
	#pragma omp parallel for simd collapse(3) schedule(static) \
		aligned(ptr_v, ptr_x, ptr_y, ptr_z, ptr_ux, ptr_uy, ptr_uz : mem_align)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				cuda::std::array<T, 3> r_vec = { ptr_x[idx], ptr_y[idx], ptr_z[idx] };
				cuda::std::array<T, 3> u_vec = { ptr_ux[idx], ptr_uy[idx], ptr_uz[idx] };
				T x = r_vec[0], y = r_vec[1];
				T ux = u_vec[0], uy = u_vec[1];
				
				T lz = m_e<T> * (x * uy - y * ux);
				
				ptr_v[idx] = lz;
			}
		}
	}
}

template <std::floating_point T>
void compute_u_field(ComplexScalarField<T> &u_field, const Laser<T> &laser) noexcept {
	int nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	T r_max_x = u_field.r_max[0], r_max_y = u_field.r_max[1], r_max_z = u_field.r_max[2], z_r = laser.z_r, w0 = laser.w0;
	ComplexScalarFieldView u_field_view = u_field.get_cpu_view();
	cuda::std::complex<T> *__restrict__ ptr_v = std::assume_aligned<mem_align>(u_field_view.v);
	
	#pragma omp parallel for simd collapse(3) schedule(static) \
		aligned(ptr_v : mem_align)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				cuda::std::array<T, 3> r_vec = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				T z = r_vec[2];
				T r_z = compute_r_z(z, z_r);
				T w_z = compute_w_z(w0, z, z_r);
				
				cuda::std::complex<T> u_i = compute_u(laser, r_vec, r_z, w_z);
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				
				ptr_v[idx] = u_i;
			}
		}
	}
}

template <std::floating_point T>
void compute_eb_field(VectorField<T> &e_field, VectorField<T> &b_field, const ComplexScalarField<T> &u_field, const Laser<T> &laser, T t) noexcept {
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	T r_max_x = e_field.r_max[0], r_max_y = e_field.r_max[1], r_max_z = e_field.r_max[2];
	ComplexScalarFieldView u_field_view = u_field.get_cpu_view();
	VectorFieldView e_field_view = e_field.get_cpu_view(), b_field_view = b_field.get_cpu_view();
	T *__restrict__ ptr_x_e = std::assume_aligned<mem_align>(e_field_view.x);
	T *__restrict__ ptr_y_e = std::assume_aligned<mem_align>(e_field_view.y);
	T *__restrict__ ptr_z_e = std::assume_aligned<mem_align>(e_field_view.z);
	T *__restrict__ ptr_x_b = std::assume_aligned<mem_align>(b_field_view.x);
	T *__restrict__ ptr_y_b = std::assume_aligned<mem_align>(b_field_view.y);
	T *__restrict__ ptr_z_b = std::assume_aligned<mem_align>(b_field_view.z);
	cuda::std::complex<T> *__restrict__ ptr_v = std::assume_aligned<mem_align>(u_field_view.v);
	
	#pragma omp parallel for simd collapse(3) schedule(static) \
		aligned(ptr_x_e, ptr_y_e, ptr_z_e, ptr_x_b, ptr_y_b, ptr_z_b, ptr_v : mem_align)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				cuda::std::array<T, 3> r_vec = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				
				EBVectors<T> eb_vec = compute_eb(ptr_v, laser, r_vec, t, idx);
				
				ptr_x_e[idx] = eb_vec.e[0]; ptr_y_e[idx] = eb_vec.e[1]; ptr_z_e[idx] = eb_vec.e[2];
				ptr_x_b[idx] = eb_vec.b[0]; ptr_y_b[idx] = eb_vec.b[1]; ptr_z_b[idx] = eb_vec.b[2];
			}
		}
	}
}

template void compute_lz<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void compute_u_field<double>(ComplexScalarField<double> &u_field, const Laser<double> &laser) noexcept;
template void compute_eb_field<double>(VectorField<double> &e_field, VectorField<double> &b_field, const ComplexScalarField<double> &u_field, const Laser<double> &laser, double t) noexcept;
 
template void compute_lz<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void compute_u_field<float>(ComplexScalarField<float> &u_field, const Laser<float> &laser) noexcept;
template void compute_eb_field<float>(VectorField<float> &e_field, VectorField<float> &b_field,  const ComplexScalarField<float> &u_field, const Laser<float> &laser, float t) noexcept;