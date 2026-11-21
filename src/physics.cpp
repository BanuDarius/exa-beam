// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <cuda/std/cmath>
#include <cuda/std/array>

#include "physics.hpp"
#include "math_functions.hpp"
#include "laguerre_gauss.hpp"

template <std::floating_point T>
void compute_lz(ScalarField<T> &lz_data, const Particles<T> &particles_data) noexcept {
	ScalarFieldView lz = lz_data.get_cpu_view();
	ParticlesView particles = particles_data.get_cpu_view();
	int nx = lz.num[0], ny = lz.num[1], nz = lz.num[2];
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
			for(int i = 0; i < nx; i++) {
				std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
				Vector3<T> r_vec = particles.get_position(idx);
				Vector3<T> u_vec = particles.get_velocity(idx);
				T x = r_vec[0], y = r_vec[1];
				T ux = u_vec[0], uy = u_vec[1];
				
				T lz_s = m_e<T> * (x * uy - y * ux);
				
				lz.set_field(lz_s, idx);
			}
}

template <std::floating_point T>
void fdtd_update_b(VectorField<T> &b_data, const VectorField<T> &e_data, T dt) noexcept {
	VectorFieldView b = b_data.get_cpu_view(), e = e_data.get_cpu_view();
	
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
				
				T curl_Ex = dyEz - dzEy;
				T curl_Ey = dzEx - dxEz;
				T curl_Ez = dxEy - dyEx;
				
				b.set_x(b.get_x(i, j, k) - dt * curl_Ex, i, j, k);
				b.set_y(b.get_y(i, j, k) - dt * curl_Ey, i, j, k);
				b.set_z(b.get_z(i, j, k) - dt * curl_Ez, i, j, k);
			}
}

template <std::floating_point T>
void fdtd_update_e(VectorField<T> &e_data, const VectorField<T> &b_data, const VectorField<T> &j_data, T dt) noexcept {
	VectorFieldView e = e_data.get_cpu_view(), b = b_data.get_cpu_view(), J = j_data.get_cpu_view();
	
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2];
	
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 0; i < nx - 1; i++) {
				T dyBz = inv_dy * (b.get_z(i, j, k) - b.get_z(i, j - 1, k));
				T dzBy = inv_dz * (b.get_y(i, j, k) - b.get_y(i, j, k - 1));
				T curl_Bx = dyBz - dzBy;
				T dtEx = c2<T> * curl_Bx - T(4.0) * pi<T> * J.get_x(i, j, k);
				
				e.set_x(e.get_x(i, j, k) + dt * dtEx, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 0; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dzBx = inv_dz * (b.get_x(i, j, k) - b.get_x(i, j, k - 1));
				T dxBz = inv_dx * (b.get_z(i, j, k) - b.get_z(i - 1, j, k));
				T curl_By = dzBx - dxBz;
				T dtEy = c2<T> * curl_By - T(4.0) * pi<T> * J.get_y(i, j, k);
				
				e.set_y(e.get_y(i, j, k) + dt * dtEy, i, j, k);
			}
	#pragma omp parallel for simd collapse(3) schedule(static)
	for(int k = 0; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T dxBy = inv_dx * (b.get_y(i, j, k) - b.get_y(i - 1, j, k));
				T dyBx = inv_dy * (b.get_x(i, j, k) - b.get_x(i, j - 1, k));
				T curl_Bz = dxBy - dyBx;
				T dtEz = c2<T> * curl_Bz - T(4.0) * pi<T> * J.get_z(i, j, k);
				
				e.set_z(e.get_z(i, j, k) + dt * dtEz, i, j, k);
			}
}

template <std::floating_point T>
T total_energy(const VectorField<T> &e_data, const VectorField<T> &b_data) noexcept {
	VectorFieldView e = e_data.get_cpu_view(), b = b_data.get_cpu_view();
	
	T energy = T(0.0);
	T dV = e.dx[0] * e.dx[1] * e.dx[2];
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	#pragma omp parallel for simd collapse(3) schedule(static) reduction(+:energy)
	for(int k = 0; k < nz; k++)
		for(int j = 0; j < ny; j++)
			for(int i = 0; i < nx; i++) {
				Vector3<T> e_vec = e.get_vector(i, j, k), b_vec = b.get_vector(i, j, k);
				energy += T(0.5) * (dot(e_vec, e_vec) + c2<T> * dot(b_vec, b_vec));
			}
	return energy * dV;
}

template <std::floating_point T>
T divergence_e(const VectorField<T> &e_data) noexcept {
	VectorFieldView e = e_data.get_cpu_view();
	
	T divergence = T(0.0);
	int nx = e.num[0], ny = e.num[1], nz = e.num[2], num_points = (nx - 2) * (ny - 2) * (nz - 2);
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2];
	#pragma omp parallel for simd collapse(3) schedule(static) reduction(+:divergence)
	for(int k = 1; k < nz - 1; k++)
		for(int j = 1; j < ny - 1; j++)
			for(int i = 1; i < nx - 1; i++) {
				T e_x = e.get_x(i, j, k);
				T e_y = e.get_y(i, j, k);
				T e_z = e.get_z(i, j, k);
				
				T dxEx = inv_dx * (e_x - e.get_x(i - 1, j, k));
				T dyEy = inv_dy * (e_y - e.get_y(i, j - 1, k));
				T dzEz = inv_dz * (e_z - e.get_z(i, j, k - 1));
				
				T divE = dxEx + dyEy + dzEz;
				divergence += divE * divE;
			}
	return cuda::std::sqrt(divergence / static_cast<T>(num_points));
}

template <std::floating_point T>
void initialize_ebj(VectorField<T> &e_data, VectorField<T> &b_data, VectorField<T> &j_data, std::span<const Laser<T>> lasers, T dt) noexcept {
	VectorFieldView e = e_data.get_cpu_view(), b = b_data.get_cpu_view(), J = j_data.get_cpu_view();
	
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
				
				Vector3<T> r_vec_global = { x + dx * T(0.5), y, z };
				T e_x = T(0.0);
				if(i < nx - 1 && j < ny - 1 && j > 0 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_x += lasers[q].vec_local_to_global(e_vec_local)[0];
					}
				e.set_x(e_x, i, j, k);
				J.set_x(T(0.0), i, j, k);
				
				r_vec_global = { x, y + dy * T(0.5), z };
				T e_y = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_y += lasers[q].vec_local_to_global(e_vec_local)[1];
					}
				e.set_y(e_y, i, j, k);
				J.set_y(T(0.0), i, j, k);
				
				r_vec_global = { x, y, z + dz * T(0.5) };
				T e_z = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && j > 0 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
						e_z += lasers[q].vec_local_to_global(e_vec_local)[2];
					}
				e.set_z(e_z, i, j, k);
				J.set_z(T(0.0), i, j, k);
				
				r_vec_global = { x, y + dy * T(0.5), z + dz * T(0.5) };
				T b_x = T(0.0);
				if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_x += lasers[q].vec_local_to_global(b_vec_local)[0];
					}
				b.set_x(b_x, i, j, k);
				
				r_vec_global = { x + dx * T(0.5), y, z + dz * T(0.5) };
				T b_y = T(0.0);
				if(i < nx - 1 && j < ny - 1 && j > 0 && k < nz - 1)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_y += lasers[q].vec_local_to_global(b_vec_local)[1];
					}
				b.set_y(b_y, i, j, k);
				
				r_vec_global = { x + dx * T(0.5), y + dy * T(0.5), z };
				T b_z = T(0.0);
				if(i < nx - 1 && j < ny - 1 && k < nz - 1 && k > 0)
					for(int q = 0; q < num_lasers; q++) {
						Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
						Vector3<T> b_vec_local = lg_compute_b(lasers[q], r_vec_local, - T(0.5) * dt);
						b_z += lasers[q].vec_local_to_global(b_vec_local)[2];
					}
				b.set_z(b_z, i, j, k);
			}
}

template void compute_lz<double>(ScalarField<double> &lz_field, const Particles<double> &particles) noexcept;
template void fdtd_update_b<double>(VectorField<double> &b_data, const VectorField<double> &e_data, double dt) noexcept;
template void fdtd_update_e<double>(VectorField<double> &e_data, const VectorField<double> &b_data, const VectorField<double> &j_data, double dt) noexcept;
template double total_energy<double>(const VectorField<double> &e_data, const VectorField<double> &b_data) noexcept;
template double divergence_e<double>(const VectorField<double> &e_data) noexcept;
template void initialize_ebj<double>(VectorField<double> &e_data, VectorField<double> &b_data, VectorField<double> &j_data, std::span<const Laser<double>> lasers, double dt) noexcept;

template void compute_lz<float>(ScalarField<float> &lz_field, const Particles<float> &particles) noexcept;
template void fdtd_update_b<float>(VectorField<float> &b_data, const VectorField<float> &e_data, float dt) noexcept;
template void fdtd_update_e<float>(VectorField<float> &e_data, const VectorField<float> &b_data, const VectorField<float> &j_data, float dt) noexcept;
template float total_energy<float>(const VectorField<float> &e_data, const VectorField<float> &b_data) noexcept;
template float divergence_e<float>(const VectorField<float> &e_data) noexcept;
template void initialize_ebj<float>(VectorField<float> &e_data, VectorField<float> &b_data, VectorField<float> &j_data, std::span<const Laser<float>> lasers, float dt) noexcept;
