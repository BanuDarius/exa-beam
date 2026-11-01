// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/complex>

#include "physics_gpu.hpp"
#include "math_functions.hpp"
#include "laguerre_gauss.hpp"

template <std::floating_point T>
__global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, ParticlesView<T> particles_view) {
	std::size_t particle_num = particles_view.particle_num;
	std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < particle_num) {
		cuda::std::array<T, 3> r_vec = particles_view.get_position(idx);
		cuda::std::array<T, 3> u_vec = particles_view.get_velocity(idx);
		T x = r_vec[0], y = r_vec[1];
		T ux = u_vec[0], uy = u_vec[1];
		
		T lz = m_e<T> * (x * uy - y * ux);
		
		lz_view.set_field(lz, idx);
	}
}

template <std::floating_point T>
__global__ void fdtd_update_b_gpu_kernel(VectorFieldView<T> b, VectorFieldView<T> e, T dt) {
	int nx = b.num[0], ny = b.num[1], nz = b.num[2];
	T inv_dx = b.inv_dx[0], inv_dy = b.inv_dx[1], inv_dz = b.inv_dx[2];
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < nx - 1 && j < ny - 1 && k < nz - 1) {
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
__global__ void fdtd_update_e_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, T dt) {
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2], c2 = c<T> * c<T>;
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < nx - 1 && j < ny - 1 && j > 0 && k < nz - 1 && k > 0) {
		T dyBz = inv_dy * (b.get_z(i, j, k) - b.get_z(i, j - 1, k));
		T dzBy = inv_dz * (b.get_y(i, j, k) - b.get_y(i, j, k - 1));
		T dtEx = c2 * (dyBz - dzBy);
		
		e.set_x(e.get_x(i, j, k) + dt * dtEx, i, j, k);
	}
	if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1 && k > 0) {
		T dzBx = inv_dz * (b.get_x(i, j, k) - b.get_x(i, j, k - 1));
		T dxBz = inv_dx * (b.get_z(i, j, k) - b.get_z(i - 1, j, k));
		T dtEy = c2 * (dzBx - dxBz);
		
		e.set_y(e.get_y(i, j, k) + dt * dtEy, i, j, k);
	}
	if(i < nx - 1 && i > 0 && j < ny - 1 && j > 0 && k < nz - 1) {
		T dxBy = inv_dx * (b.get_y(i, j, k) - b.get_y(i - 1, j, k));
		T dyBx = inv_dy * (b.get_x(i, j, k) - b.get_x(i, j - 1, k));
		T dtEz = c2 * (dxBy - dyBx);
		
		e.set_z(e.get_z(i, j, k) + dt * dtEz, i, j, k);
	}
}

template <std::floating_point T>
void compute_lz_gpu(ScalarField<T> &lz_field, Particles<T> &particles) noexcept {
	std::size_t particle_num = particles.particle_num;
	dim3 threads(threads_1d_nx);
	dim3 blocks((particle_num + threads.x - 1) / threads.x);
	
	ScalarFieldView<T> lz_view = lz_field.get_gpu_view();
	ParticlesView<T> particles_view = particles.get_gpu_view();
	compute_lz_gpu_kernel<<<blocks, threads>>>(lz_view, particles_view);
	CUDA_CHECK(cudaGetLastError());
}

template <std::floating_point T>
void fdtd_update_b_gpu(VectorField<T> &b_field, VectorField<T> &e_field, T dt) noexcept {
	VectorFieldView<T> b = b_field.get_gpu_view(), e = e_field.get_gpu_view();
	int nx = b_field.num[0], ny = b_field.num[1], nz = b_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	dim3 blocks(
		(nx + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nz + threads.z - 1) / threads.z
	);
	
	fdtd_update_b_gpu_kernel<<<blocks, threads>>>(b, e, dt);
	CUDA_CHECK(cudaGetLastError());
}

template <std::floating_point T>
void fdtd_update_e_gpu(VectorField<T> &e_field, VectorField<T> &b_field, T dt) noexcept {
	VectorFieldView<T> b = b_field.get_gpu_view(), e = e_field.get_gpu_view();
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	dim3 blocks(
		(nx + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nz + threads.z - 1) / threads.z
	);
	
	fdtd_update_e_gpu_kernel<<<blocks, threads>>>(b, e, dt);
	CUDA_CHECK(cudaGetLastError());
}

template __global__ void compute_lz_gpu_kernel<double>(ScalarFieldView<double> lz_view, ParticlesView<double> particles_view);
template __global__ void fdtd_update_b_gpu_kernel<double>(VectorFieldView<double> b, VectorFieldView<double> e, double dt);
template __global__ void fdtd_update_e_gpu_kernel<double>(VectorFieldView<double> e, VectorFieldView<double> b, double dt);
template void compute_lz_gpu<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void fdtd_update_b_gpu<double>(VectorField<double> &b_field, VectorField<double> &e_field, double dt) noexcept;
template void fdtd_update_e_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field, double dt) noexcept;

template __global__ void compute_lz_gpu_kernel<float>(ScalarFieldView<float> lz_view, ParticlesView<float> particles_view);
template __global__ void fdtd_update_b_gpu_kernel<float>(VectorFieldView<float> b, VectorFieldView<float> e, float dt);
template __global__ void fdtd_update_e_gpu_kernel<float>(VectorFieldView<float> e, VectorFieldView<float> b, float dt);
template void compute_lz_gpu<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void fdtd_update_b_gpu<float>(VectorField<float> &b_field, VectorField<float> &e_field, float dt) noexcept;
template void fdtd_update_e_gpu<float>(VectorField<float> &e_field, VectorField<float> &b_field, float dt) noexcept;