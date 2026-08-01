// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <span>

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
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				cuda::std::array<T, 3> r_vec = particles_view.get_position(idx);
				cuda::std::array<T, 3> u_vec = particles_view.get_velocity(idx);
				T x = r_vec[0], y = r_vec[1];
				T ux = u_vec[0], uy = u_vec[1];
				
				T lz = m_e<T> * (x * uy - y * ux);
				
				lz_field_view.set_field(lz, idx);
			}
		}
	}
}

template <std::floating_point T>
void compute_u_field(ComplexScalarField<T> &u_field, std::span<const Laser<T>> lasers) noexcept {
	int laser_count = lasers.front().laser_count, nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	T r_max_x = u_field.r_max[0], r_max_y = u_field.r_max[1], r_max_z = u_field.r_max[2];
	ComplexScalarFieldView u_field_view = u_field.get_cpu_view();
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				cuda::std::array<T, 3> r_vec_global = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				cuda::std::complex<T> u_i{};
				for(int q = 0; q < laser_count; q++) {
					cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
					
					T z = r_vec_local[2];
					T z_r = lasers[q].z_r, w0 = lasers[q].w0;
					T r_z = compute_r_z(z, z_r);
					T w_z = compute_w_z(w0, z, z_r);
					
					u_i += compute_u(lasers[q], r_vec_local, r_z, w_z);
				}
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				u_field_view.set_field(u_i, idx);
			}
		}
	}
}

template <std::floating_point T>
void compute_eb_field(VectorField<T> &e_field, VectorField<T> &b_field, std::span<const Laser<T>> lasers, T t) noexcept {
	int laser_count = lasers.front().laser_count, nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	T r_max_x = e_field.r_max[0], r_max_y = e_field.r_max[1], r_max_z = e_field.r_max[2];
	VectorFieldView e_field_view = e_field.get_cpu_view(), b_field_view = b_field.get_cpu_view();
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int i = 0; i < nx; i++) {
		for(int j = 0; j < ny; j++) {
			for(int k = 0; k < nz; k++) {
				cuda::std::array<T, 3> r_vec_global = {
					interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
					interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
					interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
				};
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				EBVectors<T> eb_vec_global;
				for(int q = 0; q < laser_count; q++) {
					cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
					EBVectors<T> eb_vec_local = compute_eb(lasers[q], r_vec_local, t);
					eb_vec_global = EBVectors(
						eb_vec_global.e + lasers[q].vec_local_to_global(eb_vec_local.e),
						eb_vec_global.b + lasers[q].vec_local_to_global(eb_vec_local.b)
					);
				}
				e_field_view.set_field(eb_vec_global.e, idx);
				b_field_view.set_field(eb_vec_global.b, idx);
			}
		}
	}
}

template void compute_lz<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void compute_u_field<double>(ComplexScalarField<double> &u_field, std::span<const Laser<double>> lasers) noexcept;
template void compute_eb_field<double>(VectorField<double> &e_field, VectorField<double> &b_field, std::span<const Laser<double>> lasers, double t) noexcept;
 
template void compute_lz<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void compute_u_field<float>(ComplexScalarField<float> &u_field, std::span<const Laser<float>> lasers) noexcept;
template void compute_eb_field<float>(VectorField<float> &e_field, VectorField<float> &b_field, std::span<const Laser<float>> lasers, float t) noexcept;