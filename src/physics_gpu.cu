// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <array>
#include <complex>

#include <cuda_runtime.h>
#include <cuda/std/array>

#include "physics_gpu.hpp"
#include "laguerre_gauss.hpp"
#include "math_functions.hpp"

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
__global__ void compute_u_field_gpu_kernel(ComplexScalarFieldView<T> u_field_view, __grid_constant__ const GPULasers<T> lasers) {
	T r_max_x = u_field_view.r_max[0], r_max_y = u_field_view.r_max[1], r_max_z = u_field_view.r_max[2];
	int laser_count = lasers.laser_count, nx = u_field_view.num[0], ny = u_field_view.num[1], nz = u_field_view.num[2];
	std::size_t field_size = u_field_view.field_size;
	
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < nx && j < ny && k < nz) {
		std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
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
		
		u_field_view.set_field(u_i, idx);
	}
}

template <std::floating_point T>
__global__ void compute_eb_field_gpu_kernel(VectorFieldView<T> e_field_view, VectorFieldView<T> b_field_view, __grid_constant__ const GPULasers<T> lasers, T t) {
	T r_max_x = e_field_view.r_max[0], r_max_y = e_field_view.r_max[1], r_max_z = e_field_view.r_max[2];
	int laser_count = lasers.laser_count, nx = e_field_view.num[0], ny = e_field_view.num[1], nz = e_field_view.num[2];
	std::size_t field_size = e_field_view.field_size;
	
	int i = blockIdx.z * blockDim.z + threadIdx.z;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < nx && j < ny && k < nz) {
		std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
		cuda::std::array<T, 3> r_vec_global = {
			interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
			interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
			interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
		};
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
void compute_u_field_gpu(ComplexScalarField<T> &u_field, const GPULasers<T> &lasers) noexcept {
	int nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	dim3 blocks(
		(nz + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nx + threads.z - 1) / threads.z
	);
	
	ComplexScalarFieldView<T> u_field_view = u_field.get_gpu_view();
	compute_u_field_gpu_kernel<<<blocks, threads>>>(u_field_view, lasers);
	CUDA_CHECK(cudaGetLastError());
}

template <std::floating_point T>
void compute_eb_field_gpu(VectorField<T> &e_field, VectorField<T> &b_field, const GPULasers<T> &lasers, T t) noexcept {
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	dim3 blocks(
		(nz + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nx + threads.z - 1) / threads.z
	);
	
	VectorFieldView<T> e_field_view = e_field.get_gpu_view();
	VectorFieldView<T> b_field_view = b_field.get_gpu_view();
	compute_eb_field_gpu_kernel<<<blocks, threads>>>(e_field_view, b_field_view, lasers, t);
	CUDA_CHECK(cudaGetLastError());
}
template __global__ void compute_lz_gpu_kernel<double>(ScalarFieldView<double> lz_view, ParticlesView<double> particles_view);
template __global__ void compute_u_field_gpu_kernel<double>(ComplexScalarFieldView<double> u_field_view, __grid_constant__ const GPULasers<double> lasers);
template __global__ void compute_eb_field_gpu_kernel<double>(VectorFieldView<double> e_field_view, VectorFieldView<double> b_field_view, __grid_constant__ const GPULasers<double> lasers, double t);
template void compute_lz_gpu<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void compute_u_field_gpu<double>(ComplexScalarField<double> &u_field, const GPULasers<double> &lasers) noexcept;
template void compute_eb_field_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field, const GPULasers<double> &lasers, double t) noexcept;

template __global__ void compute_lz_gpu_kernel<float>(ScalarFieldView<float> lz_view, ParticlesView<float> particles_view);
template __global__ void compute_u_field_gpu_kernel<float>(ComplexScalarFieldView<float> u_field_view, __grid_constant__ const GPULasers<float> lasers);
template __global__ void compute_eb_field_gpu_kernel<float>(VectorFieldView<float> e_field_view, VectorFieldView<float> b_field_view, __grid_constant__ const GPULasers<float> lasers, float t);
template void compute_lz_gpu<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void compute_u_field_gpu<float>(ComplexScalarField<float> &u_field, const GPULasers<float> &lasers) noexcept;
template void compute_eb_field_gpu<float>(VectorField<float> &e_field, VectorField<float> &b_field, const GPULasers<float> &lasers, float t) noexcept;