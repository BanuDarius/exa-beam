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
		Vector3<T> r_vec = particles_view.get_position(idx);
		Vector3<T> u_vec = particles_view.get_velocity(idx);
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
__global__ void total_energy_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, T *d_energy) {
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	extern __shared__ char shared_memory[];
	T *shared_energy = reinterpret_cast<T*>(shared_memory);
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	int local_idx = grid_idx(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
	
	T energy = T(0.0);
	if(i < nx && j < ny && k < nz) {
		Vector3<T> e_vec = e.get_vector(i, j, k), b_vec = b.get_vector(i, j, k);
		energy = T(0.5) * (dot(e_vec, e_vec) + c2<T> * dot(b_vec, b_vec));
	}
	shared_energy[local_idx] = energy;
	__syncthreads();
	
	for(int iter_idx = block_size / 2; iter_idx > 0; iter_idx /= 2) {
		if(local_idx < iter_idx) shared_energy[local_idx] += shared_energy[local_idx + iter_idx];
		__syncthreads();
	}
	
	if(local_idx == 0)
		atomicAdd(d_energy, shared_energy[0]);
}

template <std::floating_point T>
__global__ void divergence_e_gpu_kernel(VectorFieldView<T> e, T *d_divergence) {
	int nx = e.num[0], ny = e.num[1], nz = e.num[2];
	T inv_dx = e.inv_dx[0], inv_dy = e.inv_dx[1], inv_dz = e.inv_dx[2];
	extern __shared__ char shared_memory[];
	T *shared_divergence = reinterpret_cast<T*>(shared_memory);
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	int block_size = blockDim.x * blockDim.y * blockDim.z;
	int local_idx = grid_idx(threadIdx.x, threadIdx.y, threadIdx.z, blockDim.x, blockDim.y, blockDim.z);
	
	T divergence = T(0.0);
	if(i < nx - 1 && i > 0 && j < ny - 1 && j > 0 && k < nz - 1 && k > 0) {
		T e_x = e.get_x(i, j, k);
		T e_y = e.get_y(i, j, k);
		T e_z = e.get_z(i, j, k);
		
		T dxEx = inv_dx * (e_x - e.get_x(i - 1, j, k));
		T dyEy = inv_dy * (e_y - e.get_y(i, j - 1, k));
		T dzEz = inv_dz * (e_z - e.get_z(i, j, k - 1));
		
		T divE = dxEx + dyEy + dzEz;
		divergence += divE * divE;
	}
	shared_divergence[local_idx] = divergence;
	__syncthreads();
	
	for(int iter_idx = block_size / 2; iter_idx > 0; iter_idx /= 2) {
		if(local_idx < iter_idx) shared_divergence[local_idx] += shared_divergence[local_idx + iter_idx];
		__syncthreads();
	}
	
	if(local_idx == 0)
		atomicAdd(d_divergence, shared_divergence[0]);
}

template <std::floating_point T>
__global__ void initialize_eb_fields_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, __grid_constant__ const DeviceLasers<T> lasers, T dt) {
	int nx = e.num[0], ny = e.num[1], nz = e.num[2], num_lasers = lasers.laser_count;
	T r_max_x = e.r_max[0], r_max_y = e.r_max[1], r_max_z = e.r_max[2];
	T dx = T(2.0) * r_max_x / (nx - 1), dy = T(2.0) * r_max_y / (ny - 1), dz = T(2.0) * r_max_z / (nz - 1);
	
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < nx && j < ny && k < nz) {
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
		
		r_vec_global = { x, y + dy * T(0.5), z };
		T e_y = T(0.0);
		if(i < nx - 1 && i > 0 && j < ny - 1 && k < nz - 1 && k > 0)
			for(int q = 0; q < num_lasers; q++) {
				Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
				Vector3<T> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
				e_y += lasers[q].vec_local_to_global(e_vec_local)[1];
			}
		e.set_y(e_y, i, j, k);
		
		r_vec_global = { x, y, z + dz * T(0.5) };
		T e_z = T(0.0);
		if(i < nx - 1 && i > 0 && j < ny - 1 && j > 0 && k < nz - 1)
			for(int q = 0; q < num_lasers; q++) {
				Vector3<T> r_vec_local = lasers[q].pos_global_to_local(r_vec_global);
				Vector3<T> e_vec_local = lg_compute_e(lasers[q], r_vec_local, T(0.0));
				e_z += lasers[q].vec_local_to_global(e_vec_local)[2];
			}
		e.set_z(e_z, i, j, k);
		
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
	
	fdtd_update_e_gpu_kernel<<<blocks, threads>>>(e, b, dt);
	CUDA_CHECK(cudaGetLastError());
}

template <std::floating_point T>
T total_energy_gpu(VectorField<T> &e_field, VectorField<T> &b_field) noexcept {
	VectorFieldView<T> b = b_field.get_gpu_view(), e = e_field.get_gpu_view();
	T dV = e.dx[0] * e.dx[1] * e.dx[2];
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	int shared_memory = threads.x * threads.y * threads.z * sizeof(T);
	dim3 blocks(
		(nx + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nz + threads.z - 1) / threads.z
	);
	
	T *d_energy = nullptr, h_energy;
	CUDA_CHECK(cudaMalloc(&d_energy, sizeof(T)));
	CUDA_CHECK(cudaMemset(d_energy, 0, sizeof(T)));
	
	total_energy_gpu_kernel<<<blocks, threads, shared_memory>>>(e, b, d_energy);
	CUDA_CHECK(cudaGetLastError());
	
	CUDA_CHECK(cudaMemcpy(&h_energy, d_energy, sizeof(T), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_energy));
	return h_energy * dV;
}

template <std::floating_point T>
T divergence_e_gpu(VectorField<T> &e_field) noexcept {
	VectorFieldView<T> e = e_field.get_gpu_view();
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2], num_points = (nx - 2) * (ny - 2) * (nz - 2);
	
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	int shared_memory = threads.x * threads.y * threads.z * sizeof(T);
	dim3 blocks(
		(nx + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nz + threads.z - 1) / threads.z
	);
	
	T *d_divergence, h_divergence;
	CUDA_CHECK(cudaMalloc(&d_divergence, sizeof(T)));
	CUDA_CHECK(cudaMemset(d_divergence, 0, sizeof(T)));
	
	divergence_e_gpu_kernel<<<blocks, threads, shared_memory>>>(e, d_divergence);
	CUDA_CHECK(cudaGetLastError());
	
	CUDA_CHECK(cudaMemcpy(&h_divergence, d_divergence, sizeof(T), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaFree(d_divergence));
	return cuda::std::sqrt(h_divergence / static_cast<T>(num_points));
}

template <std::floating_point T>
void initialize_eb_fields_gpu(VectorField<T> &e_field, VectorField<T> &b_field, const DeviceLasers<T> &lasers, T dt) noexcept {
	VectorFieldView<T> e = e_field.get_gpu_view(), b = b_field.get_gpu_view();
	int nx = e_field.num[0], ny = e_field.num[1], nz = e_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_ny, threads_3d_nz);
	dim3 blocks(
		(nx + threads.x - 1) / threads.x,
		(ny + threads.y - 1) / threads.y,
		(nz + threads.z - 1) / threads.z
	);
	
	initialize_eb_fields_gpu_kernel<<<blocks, threads>>>(e, b, lasers, dt);
	CUDA_CHECK(cudaGetLastError());
}

template __global__ void compute_lz_gpu_kernel<double>(ScalarFieldView<double> lz_view, ParticlesView<double> particles_view);
template __global__ void fdtd_update_b_gpu_kernel<double>(VectorFieldView<double> b, VectorFieldView<double> e, double dt);
template __global__ void fdtd_update_e_gpu_kernel<double>(VectorFieldView<double> e, VectorFieldView<double> b, double dt);
template __global__ void total_energy_gpu_kernel<double>(VectorFieldView<double> e, VectorFieldView<double> b, double *d_energy);
template __global__ void divergence_e_gpu_kernel<double>(VectorFieldView<double> e, double *d_divergence);
template __global__ void initialize_eb_fields_gpu_kernel<double>(VectorFieldView<double> e, VectorFieldView<double> b, __grid_constant__ const DeviceLasers<double> lasers, double dt);
template void compute_lz_gpu<double>(ScalarField<double> &lz_field, Particles<double> &particles) noexcept;
template void fdtd_update_b_gpu<double>(VectorField<double> &b_field, VectorField<double> &e_field, double dt) noexcept;
template void fdtd_update_e_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field, double dt) noexcept;
template double total_energy_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field) noexcept;
template double divergence_e_gpu<double>(VectorField<double> &e_field) noexcept;
template void initialize_eb_fields_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field, const DeviceLasers<double> &lasers, double dt) noexcept;

template __global__ void compute_lz_gpu_kernel<float>(ScalarFieldView<float> lz_view, ParticlesView<float> particles_view);
template __global__ void fdtd_update_b_gpu_kernel<float>(VectorFieldView<float> b, VectorFieldView<float> e, float dt);
template __global__ void fdtd_update_e_gpu_kernel<float>(VectorFieldView<float> e, VectorFieldView<float> b, float dt);
template __global__ void total_energy_gpu_kernel<float>(VectorFieldView<float> e, VectorFieldView<float> b, float *d_energy);
template __global__ void divergence_e_gpu_kernel<float>(VectorFieldView<float> e, float *d_divergence);
template __global__ void initialize_eb_fields_gpu_kernel<float>(VectorFieldView<float> e, VectorFieldView<float> b, __grid_constant__ const DeviceLasers<float> lasers, float dt);
template void compute_lz_gpu<float>(ScalarField<float> &lz_field, Particles<float> &particles) noexcept;
template void fdtd_update_b_gpu<float>(VectorField<float> &b_field, VectorField<float> &e_field, float dt) noexcept;
template void fdtd_update_e_gpu<float>(VectorField<float> &e_field, VectorField<float> &b_field, float dt) noexcept;
template float total_energy_gpu(VectorField<float> &e_field, VectorField<float> &b_field) noexcept;
template float divergence_e_gpu<float>(VectorField<float> &e_field) noexcept;
template void initialize_eb_fields_gpu<float>(VectorField<float> &e_field, VectorField<float> &b_field, const DeviceLasers<float> &lasers, float dt) noexcept;