// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <cuda/std/array>
#include <cuda/std/complex>

#include "physics.hpp"
#include "math_functions.hpp"
#include "laguerre_gauss.hpp"

template <std::floating_point T>
void compute_lz(ScalarField<T> &lz_field, Particles<T> &particles) noexcept {
	int nx = particles.num[0], ny = particles.num[1], nz = particles.num[2];
	ScalarFieldView lz_field_view = lz_field.get_cpu_view();
	ParticlesView particles_view = particles.get_cpu_view();
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz; k++) {
		for(int j = 0; j < ny; j++) {
			for(int i = 0; i < nx; i++) {
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
void fdtd_update_b(VectorField<T> &b_field, VectorField<T> &e_field, T dt) noexcept {
	VectorFieldView b = b_field.get_cpu_view(), e = e_field.get_cpu_view();
	
	int nx = b.num[0], ny = b.num[1], nz = b.num[2];
	T inv_dx = b.inv_dx[0], inv_dy = b.inv_dx[1], inv_dz = b.inv_dx[2];
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz - 1; k++)
		for(int j = 0; j < ny - 1; j++)
			for(int i = 0; i < nx - 1; i++) {
				T dzEy = inv_dz * (e.get_y(i, j, k + 1) - e.get_y(i, j, k));
				T dyEz = inv_dy * (e.get_z(i, j + 1, k) - e.get_z(i, j, k));
				
				T dxEz = inv_dx * (e.get_z(i + 1, j, k) - e.get_z(i, j, k));
				T dzEx = inv_dz * (e.get_x(i, j, k + 1) - e.get_x(i, j, k));
				
				T dyEx = inv_dy * (e.get_x(i, j + 1, k) - e.get_x(i, j, k));
				T dxEy = inv_dx * (e.get_y(i + 1, j, k) - e.get_y(i, j, k));
				
				T dtBx = dzEy - dyEz;
				T dtBy = dxEz - dzEx;
				T dtBz = dyEx - dxEy;
				
				b.set_x(b.get_x(i, j, k) + dt * dtBx, i, j, k);
				b.set_y(b.get_y(i, j, k) + dt * dtBy, i, j, k);
				b.set_z(b.get_z(i, j, k) + dt * dtBz, i, j, k);
			}
}

template <std::floating_point T>
void fdtd_update_e(VectorField<T> &e_field, VectorField<T> &b_field, T dt) noexcept {
	VectorFieldView e = e_field.get_cpu_view(), b = b_field.get_cpu_view();
	
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2], c2 = c<T> * c<T>;
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 0; i < nx - 1; i++) {
				T dyBz = inv_dy * (b.get_z(i, j, k) - b.get_z(i, j - 1, k));
				T dzBy = inv_dz * (b.get_y(i, j, k) - b.get_y(i, j, k - 1));
				T dtEx = c2 * (dyBz - dzBy);
				
				e.set_x(e.get_x(i, j, k) + dt * dtEx, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 0; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dzBx = inv_dz * (b.get_x(i, j, k) - b.get_x(i, j, k - 1));
				T dxBz = inv_dx * (b.get_z(i, j, k) - b.get_z(i - 1, j, k));
				T dtEy = c2 * (dzBx - dxBz);
				
				e.set_y(e.get_y(i, j, k) + dt * dtEy, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dxBy = inv_dx * (b.get_y(i, j, k) - b.get_y(i - 1, j, k));
				T dyBx = inv_dy * (b.get_x(i, j, k) - b.get_x(i, j - 1, k));
				T dtEz = c2 * (dxBy - dyBx);
				
				e.set_z(e.get_z(i, j, k) + dt * dtEz, i, j, k);
			}
}

template void compute_lz<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void fdtd_update_b<double>(VectorField<double> &b_field, VectorField<double> &e_field, double dt) noexcept;
template void fdtd_update_e<double>(VectorField<double> &e_field, VectorField<double> &b_field, double dt) noexcept;

template void compute_lz<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void fdtd_update_b<float>(VectorField<float> &b_field, VectorField<float> &e_field, float dt) noexcept;
template void fdtd_update_e<float>(VectorField<float> &e_field, VectorField<float> &b_field, float dt) noexcept;