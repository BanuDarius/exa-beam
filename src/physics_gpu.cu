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

#include <cuda_runtime.h>
#include <cuda/std/array>

#include "physics_gpu.hpp"
#include "laguerre_gauss.hpp"
#include "math_functions.hpp"

template <std::floating_point T>
__global__ void compute_u_field_gpu_kernel(ComplexScalarFieldView<T> u_field_view, Laser<T> laser) {
	const T z_r = laser.z_r, w0 = laser.w0;
	T r_max_x = u_field_view.r_max[0], r_max_y = u_field_view.r_max[1], r_max_z = u_field_view.r_max[2];
	int nx = u_field_view.num[0], ny = u_field_view.num[1], nz = u_field_view.num[2];
	const std::size_t field_size = u_field_view.field_size;
	
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < nx && j < ny && k < nz) {
		const std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
		cuda::std::array<T, 3> r_vec = {
			interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
			interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
			interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
		};
		T z = r_vec[2];
		T r_z = compute_r_z(z, z_r);
		T w_z = compute_w_z(w0, z, z_r);
		
		cuda::std::complex<T> u_i = compute_u(laser, r_vec, r_z, w_z);
		
		u_field_view.set_field(u_i, idx);
	}
}

template <std::floating_point T>
__global__ void compute_eb_field_gpu_kernel(VectorFieldView<T> e_field_view, VectorFieldView<T> b_field_view, ComplexScalarFieldView<T> u_field_view, Laser<T> laser, T t) {
	const T z_r = laser.z_r, w0 = laser.w0;
	T r_max_x = e_field_view.r_max[0], r_max_y = e_field_view.r_max[1], r_max_z = e_field_view.r_max[2];
	int nx = e_field_view.num[0], ny = e_field_view.num[1], nz = e_field_view.num[2];
	const std::size_t field_size = e_field_view.field_size;
	
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if(i < nx && j < ny && k < nz) {
		const std::size_t idx = grid_idx(i, j, k, nx, ny, nz);
		cuda::std::array<T, 3> r_vec = {
			interpolate(-r_max_x, r_max_x, static_cast<T>(i), static_cast<T>(nx)),
			interpolate(-r_max_y, r_max_y, static_cast<T>(j), static_cast<T>(ny)),
			interpolate(-r_max_z, r_max_z, static_cast<T>(k), static_cast<T>(nz))
		};
		EBVectors<T> eb_vec = compute_eb(u_field_view, laser, r_vec, t, idx);
		
		e_field_view.set_field(eb_vec.e, idx);
		b_field_view.set_field(eb_vec.b, idx);
	}
}

template <std::floating_point T>
void compute_u_field_gpu(ComplexScalarField<T> &u_field, const Laser<T> &laser) {
	int nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_nx, threads_3d_nx);
	dim3 blocks(nx / threads.x, ny / threads.y, nz / threads.z);
	
	ComplexScalarFieldView<T> u_field_view = u_field.get_gpu_view();
	compute_u_field_gpu_kernel<<<blocks, threads>>>(u_field_view, laser);
	cudaDeviceSynchronize();
}

template <std::floating_point T>
void compute_eb_field_gpu(VectorField<T> &e_field, VectorField<T> &b_field, ComplexScalarField<T> &u_field, const Laser<T> &laser, T t) {
	int nx = u_field.num[0], ny = u_field.num[1], nz = u_field.num[2];
	dim3 threads(threads_3d_nx, threads_3d_nx, threads_3d_nx);
	dim3 blocks(nx / threads.x, ny / threads.y, nz / threads.z);
	
	VectorFieldView<T> e_field_view = e_field.get_gpu_view();
	VectorFieldView<T> b_field_view = b_field.get_gpu_view();
	ComplexScalarFieldView<T> u_field_view = u_field.get_gpu_view();
	compute_eb_field_gpu_kernel<<<blocks, threads>>>(e_field_view, b_field_view, u_field_view, laser, t);
	cudaDeviceSynchronize();
}

template void compute_u_field_gpu<double>(ComplexScalarField<double> &u_field, const Laser<double> &laser);
template void compute_eb_field_gpu<double>(VectorField<double> &e_field, VectorField<double> &b_field, ComplexScalarField<double> &u_field, const Laser<double> &laser, double t);

template void compute_u_field_gpu<float>(ComplexScalarField<float> &u_field, const Laser<float> &laser);
template void compute_eb_field_gpu<float>(VectorField<float> &e_field, VectorField<float> &b_field, ComplexScalarField<float> &u_field, const Laser<float> &laser, float t);