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
	for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
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

template <std::floating_point T>
void fdtd_update_b(VectorField<T> &b_field, VectorField<T> &e_field, T dt) noexcept {
	VectorFieldView b = b_field.get_cpu_view(), e = e_field.get_cpu_view();
	
	int nx = b.num[0], ny = b.num[1], nz = b.num[2];
	T inv_dx = b.inv_dx[0], inv_dy = b.inv_dx[1], inv_dz = b.inv_dx[2];
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz - 1; k++)
		for(int j = 0; j < ny - 1; j++)
			for(int i = 0; i < nx - 1; i++) {
				T e_x = e.get_x(i, j, k);
				T e_y = e.get_y(i, j, k);
				T e_z = e.get_z(i, j, k);
				
				T dzEy = inv_dz * (e.get_y(i, j, k + 1) - e_y);
				T dyEz = inv_dy * (e.get_z(i, j + 1, k) - e_z);
				
				T dxEz = inv_dx * (e.get_z(i + 1, j, k) - e_z);
				T dzEx = inv_dz * (e.get_x(i, j, k + 1) - e_x);
				
				T dyEx = inv_dy * (e.get_x(i, j + 1, k) - e_x);
				T dxEy = inv_dx * (e.get_y(i + 1, j, k) - e_y);
				
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
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2];
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 0; i < nx - 1; i++) {
				T dyBz = inv_dy * (b.get_z(i, j, k) - b.get_z(i, j - 1, k));
				T dzBy = inv_dz * (b.get_y(i, j, k) - b.get_y(i, j, k - 1));
				T dtEx = c2<T> * (dyBz - dzBy);
				
				e.set_x(e.get_x(i, j, k) + dt * dtEx, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 0; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dzBx = inv_dz * (b.get_x(i, j, k) - b.get_x(i, j, k - 1));
				T dxBz = inv_dx * (b.get_z(i, j, k) - b.get_z(i - 1, j, k));
				T dtEy = c2<T> * (dzBx - dxBz);
				
				e.set_y(e.get_y(i, j, k) + dt * dtEy, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dxBy = inv_dx * (b.get_y(i, j, k) - b.get_y(i - 1, j, k));
				T dyBx = inv_dy * (b.get_x(i, j, k) - b.get_x(i, j - 1, k));
				T dtEz = c2<T> * (dxBy - dyBx);
				
				e.set_z(e.get_z(i, j, k) + dt * dtEz, i, j, k);
			}
}

template <std::floating_point T>
T compute_total_energy(VectorField<T> &e_field, VectorField<T> &b_field) noexcept {
	VectorFieldView e = e_field.get_cpu_view(), b = b_field.get_cpu_view();
	
	T energy = T(0.0);
	T dV = e.dx[0] * e.dx[1] * e.dx[2];
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	#pragma omp parallel for simd collapse(3) schedule(static) reduction(+:energy)
	for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
			for(int i = 0; i < nx; i++) {
				cuda::std::array<T, 3> e_vec = e.get_vector(i, j, k), b_vec = b.get_vector(i, j, k);
				energy += T(0.5) * (dot(e_vec, e_vec) + c2<T> * dot(b_vec, b_vec));
			}
	return energy * dV;
}

template <std::floating_point T>
void initialize_eb_fields(VectorField<T> &e_field, VectorField<T> &b_field, std::span<const Laser<T>> lasers, T dt) noexcept {
	VectorFieldView e = e_field.get_cpu_view(), b = b_field.get_cpu_view();
	
	T dx = e.dx[0], dy = e.dx[1], dz = e.dx[2];
	T r_max_x = e.r_max[0], r_max_y = e.r_max[1], r_max_z = e.r_max[2];
	int nx = e.num[0], ny = e.num[1], nz = e.num[2], num_lasers = lasers.size();
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
			for(int i = 0; i < nx; i++) {
				T x = - r_max_x + static_cast<T>(i) * dx;
				T y = - r_max_y + static_cast<T>(j) * dy;
				T z = - r_max_z + static_cast<T>(k) * dz;
				
				cuda::std::array<T, 3> r_vec_global = { x + dx * T(0.5), y, z };
				T e_x = T(0.0);
				if(i < nx - 1 && j < ny - 1 && j > 0 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_x += lasers[q].vec_local_to_global(e_vec_local)[0];
					}
				e.set_x(e_x, i, j, k);
				
				r_vec_global = { x, y + dy * T(0.5), z };
				T e_y = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_y += lasers[q].vec_local_to_global(e_vec_local)[1];
					}
				e.set_y(e_y, i, j, k);
				
				r_vec_global = { x, y, z + dz * T(0.5) };
				T e_z = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && j > 0 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_z += lasers[q].vec_local_to_global(e_vec_local)[2];
					}
				e.set_z(e_z, i, j, k);
				
				r_vec_global = { x, y + dy * T(0.5), z + dz * T(0.5) };
				T b_x = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_x += lasers[q].vec_local_to_global(b_vec_local)[0];
					}
				b.set_x(b_x, i, j, k);
				
				r_vec_global = { x + dx * T(0.5), y, z + dz * T(0.5) };
				T b_y = T(0.0);
				if(i < nx - 1 && j < ny - 1 && j > 0 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_y += lasers[q].vec_local_to_global(b_vec_local)[1];
					}
				b.set_y(b_y, i, j, k);
				
				r_vec_global = { x + dx * T(0.5), y + dy * T(0.5), z };
				T b_z = T(0.0);
				if(i < nx - 1 && j < ny - 1 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						cuda::std::array<T, 3> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						cuda::std::array<T, 3> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_z += lasers[q].vec_local_to_global(b_vec_local)[2];
					}
				b.set_z(b_z, i, j, k);
			}
}

template void compute_lz<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void fdtd_update_b<double>(VectorField<double> &b_field, VectorField<double> &e_field, double dt) noexcept;
template void fdtd_update_e<double>(VectorField<double> &e_field, VectorField<double> &b_field, double dt) noexcept;
template double compute_total_energy<double>(VectorField<double> &e_field, VectorField<double> &b_field) noexcept;
template void initialize_eb_fields<double>(VectorField<double> &e_field, VectorField<double> &b_field, std::span<const Laser<double>> lasers, double dt) noexcept;

template void compute_lz<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void fdtd_update_b<float>(VectorField<float> &b_field, VectorField<float> &e_field, float dt) noexcept;
template void fdtd_update_e<float>(VectorField<float> &e_field, VectorField<float> &b_field, float dt) noexcept;
template float compute_total_energy<float>(VectorField<float> &e_field, VectorField<float> &b_field) noexcept;
template void initialize_eb_fields<float>(VectorField<float> &e_field, VectorField<float> &b_field, std::span<const Laser<float>> lasers, float dt) noexcept;