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

#ifndef STRUCTS_VIEW_H
#define STRUCTS_VIEW_H

#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/complex>

template <std::floating_point T>
struct ParticlesView {
	T *__restrict__ x, *__restrict__ y, *__restrict__ z;
	T *__restrict__ ux, *__restrict__ uy, *__restrict__ uz, *__restrict__ gamma;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max;
	std::size_t particle_num;
	__device__ __host__ inline T get_gamma(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_gamma = std::assume_aligned<mem_align>(gamma);
			return ptr_gamma[idx];
		#else
			T gamma_v = gamma[idx];
			return gamma_v;
		#endif
	}
	__device__ __host__ inline cuda::std::array<T, 3> get_position(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			cuda::std::array<T, 3> r_vec = { ptr_x[idx], ptr_y[idx], ptr_z[idx] };
			return r_vec;
		#else
			cuda::std::array<T, 3> r_vec = { x[idx], y[idx], z[idx] };
			return r_vec;
		#endif
	}
	__device__ __host__ inline cuda::std::array<T, 3> get_velocity(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_ux = std::assume_aligned<mem_align>(ux);
			T *__restrict__ ptr_uy = std::assume_aligned<mem_align>(uy);
			T *__restrict__ ptr_uz = std::assume_aligned<mem_align>(uz);
			cuda::std::array<T, 3> u_vec = { ptr_ux[idx], ptr_uy[idx], ptr_uz[idx] };
			return u_vec;
		#else
			cuda::std::array<T, 3> u_vec = { ux[idx], uy[idx], uz[idx] };
			return u_vec;
		#endif
	}
	__device__ __host__ inline void set_gamma(T gamma_v, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_gamma = std::assume_aligned<mem_align>(gamma);
			ptr_gamma[idx] = gamma_v;
		#else
			gamma[idx] = gamma_v;
		#endif
	}
	__device__ __host__ inline void set_position(const cuda::std::array<T, 3> r_vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			ptr_x[idx] = r_vec[0]; ptr_y[idx] = r_vec[1]; ptr_z[idx] = r_vec[2];
		#else
			x[idx] = r_vec[0]; y[idx] = r_vec[1]; z[idx] = r_vec[2];
		#endif
	}
	__device__ __host__ inline void set_velocity(const cuda::std::array<T, 3> u_vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_ux = std::assume_aligned<mem_align>(ux);
			T *__restrict__ ptr_uy = std::assume_aligned<mem_align>(uy);
			T *__restrict__ ptr_uz = std::assume_aligned<mem_align>(uz);
			ptr_ux[idx] = u_vec[0]; ptr_uy[idx] = u_vec[1]; ptr_uz[idx] = u_vec[2];
		#else
			ux[idx] = u_vec[0]; uy[idx] = u_vec[1]; uz[idx] = u_vec[2];
		#endif
	}
	ParticlesView(T *x_n, T *y_n, T *z_n, T *ux_n, T *uy_n, T *uz_n, T *gamma_n, cuda::std::array<int, 3> num_n, cuda::std::array<T, 3> r_max_n, std::size_t particle_num_n)
		: x(x_n), y(y_n), z(z_n), ux(ux_n), uy(uy_n), uz(uz_n), gamma(gamma_n), num(num_n), r_max(r_max_n), particle_num(particle_num_n) {}
};

template <std::floating_point T>
struct ScalarFieldView {
	T *__restrict__ v;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max;
	std::size_t field_size;
	__device__ __host__ inline T get_field(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			return ptr_v[idx];
		#else
			T v_n = v[idx];
			return v_n;
		#endif
	}
	__device__ __host__ inline void set_field(T v_n, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			ptr_v[idx] = v_n;
		#else
			v[idx] = v_n;
		#endif
	}
	ScalarFieldView(T *v_n, cuda::std::array<int, 3> num_n, cuda::std::array<T, 3> r_max_n, std::size_t field_size_n)
		: v(v_n), num(num_n), r_max(r_max_n), field_size(field_size_n) {}
};

template <std::floating_point T>
struct ComplexScalarFieldView {
	cuda::std::complex<T> *__restrict__ v;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max;
	std::size_t field_size;
	__device__ __host__ inline cuda::std::complex<T> get_field(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			cuda::std::complex<T> *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			return ptr_v[idx];
		#else
			cuda::std::complex<T> v_n = v[idx];
			return v_n;
		#endif
	}
	__device__ __host__ inline void set_field(cuda::std::complex<T> v_n, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			cuda::std::complex<T> *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			ptr_v[idx] = v_n;
		#else
			v[idx] = v_n;
		#endif
	}
	ComplexScalarFieldView(cuda::std::complex<T> *v_n, cuda::std::array<int, 3> num_n, cuda::std::array<T, 3> r_max_n, std::size_t field_size_n)
		: v(v_n), num(num_n), r_max(r_max_n), field_size(field_size_n) {}
};

template <std::floating_point T>
struct VectorFieldView {
	T *__restrict__ x, *__restrict__ y, *__restrict__ z;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max;
	std::size_t field_size;
	__device__ __host__ inline cuda::std::array<T, 3> get_field(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			cuda::std::array<T, 3> vec = { ptr_x[idx], ptr_y[idx], ptr_z[idx] };
			return vec;
		#else
			cuda::std::array<T, 3> vec = { x[idx], y[idx], z[idx] };
			return vec;
		#endif
	}
	__device__ __host__ inline void set_field(cuda::std::array<T, 3> vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			ptr_x[idx] = vec[0]; ptr_y[idx] = vec[1]; ptr_z[idx] = vec[2];
		#else
			x[idx] = vec[0]; y[idx] = vec[1]; z[idx] = vec[2];
		#endif
	}
	VectorFieldView(T *x_n, T *y_n, T *z_n, cuda::std::array<int, 3> num_n, cuda::std::array<T, 3> r_max_n, std::size_t field_size_n)
		: x(x_n), y(y_n), z(z_n), num(num_n), r_max(r_max_n), field_size(field_size_n) {}
};

#endif