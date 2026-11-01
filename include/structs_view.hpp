// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef STRUCTS_VIEW_H
#define STRUCTS_VIEW_H

#include "math_functions.hpp"

#include <memory>

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
	__device__ __host__ T get_gamma(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_gamma = std::assume_aligned<mem_align>(gamma);
			return ptr_gamma[idx];
		#else
			return gamma[idx];
		#endif
	}
	__device__ __host__ cuda::std::array<T, 3> get_position(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			return { ptr_x[idx], ptr_y[idx], ptr_z[idx] };
		#else
			return { x[idx], y[idx], z[idx] };
		#endif
	}
	__device__ __host__ cuda::std::array<T, 3> get_velocity(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_ux = std::assume_aligned<mem_align>(ux);
			T *__restrict__ ptr_uy = std::assume_aligned<mem_align>(uy);
			T *__restrict__ ptr_uz = std::assume_aligned<mem_align>(uz);
			return { ptr_ux[idx], ptr_uy[idx], ptr_uz[idx] };
		#else
			return { ux[idx], uy[idx], uz[idx] };
		#endif
	}
	__device__ __host__ void set_gamma(T gamma_v, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_gamma = std::assume_aligned<mem_align>(gamma);
			ptr_gamma[idx] = gamma_v;
		#else
			gamma[idx] = gamma_v;
		#endif
	}
	__device__ __host__ void set_position(const cuda::std::array<T, 3> r_vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			ptr_x[idx] = r_vec[0]; ptr_y[idx] = r_vec[1]; ptr_z[idx] = r_vec[2];
		#else
			x[idx] = r_vec[0]; y[idx] = r_vec[1]; z[idx] = r_vec[2];
		#endif
	}
	__device__ __host__ void set_velocity(const cuda::std::array<T, 3> u_vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_ux = std::assume_aligned<mem_align>(ux);
			T *__restrict__ ptr_uy = std::assume_aligned<mem_align>(uy);
			T *__restrict__ ptr_uz = std::assume_aligned<mem_align>(uz);
			ptr_ux[idx] = u_vec[0]; ptr_uy[idx] = u_vec[1]; ptr_uz[idx] = u_vec[2];
		#else
			ux[idx] = u_vec[0]; uy[idx] = u_vec[1]; uz[idx] = u_vec[2];
		#endif
	}
	ParticlesView(T *x_new, T *y_new, T *z_new, T *ux_new, T *uy_new, T *uz_new, T *gamma_new, cuda::std::array<int, 3> num_new, cuda::std::array<T, 3> r_max_new, std::size_t particle_num_new)
		: x(x_new), y(y_new), z(z_new), ux(ux_new), uy(uy_new), uz(uz_new), gamma(gamma_new), num(num_new), r_max(r_max_new), particle_num(particle_num_new) {}
};

template <std::floating_point T>
struct ScalarFieldView {
	T *__restrict__ v;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max;
	std::size_t field_size;
	__device__ __host__ T get_field(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			return ptr_v[idx];
		#else
			return v[idx];
		#endif
	}
	__device__ __host__ void set_field(T v_new, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_v = std::assume_aligned<mem_align>(v);
			ptr_v[idx] = v_new;
		#else
			v[idx] = v_new;
		#endif
	}
	ScalarFieldView(T *v_new, cuda::std::array<int, 3> num_new, cuda::std::array<T, 3> r_max_new, std::size_t field_size_new)
		: v(v_new), num(num_new), r_max(r_max_new), field_size(field_size_new) {}
};

template <std::floating_point T>
struct VectorFieldView {
	T *__restrict__ x, *__restrict__ y, *__restrict__ z;
	cuda::std::array<int, 3> num;
	cuda::std::array<T, 3> r_max, inv_dx;
	std::size_t field_size;
	__device__ __host__ T get_x(int i, int j, int k) const noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			return ptr_x[idx];
		#else
			return x[idx];
		#endif
	}
	__device__ __host__ T get_y(int i, int j, int k) const noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			return ptr_y[idx];
		#else
			return y[idx];
		#endif
	}
	__device__ __host__ T get_z(int i, int j, int k) const noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			return ptr_z[idx];
		#else
			return z[idx];
		#endif
	}
	__device__ __host__ void set_x(T x_new, int i, int j, int k) noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			ptr_x[idx] = x_new;
		#else
			x[idx] = x_new;
		#endif
	}
	__device__ __host__ void set_y(T y_new, int i, int j, int k) noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			ptr_y[idx] = y_new;
		#else
			y[idx] = y_new;
		#endif
	}
	__device__ __host__ void set_z(T z_new, int i, int j, int k) noexcept {
		std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			ptr_z[idx] = z_new;
		#else
			z[idx] = z_new;
		#endif
	}
	__device__ __host__ cuda::std::array<T, 3> get_field(std::size_t idx) const noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			return { ptr_x[idx], ptr_y[idx], ptr_z[idx] };
		#else
			return { x[idx], y[idx], z[idx] };
		#endif
	}
	__device__ __host__ void set_field(cuda::std::array<T, 3> vec, std::size_t idx) noexcept {
		#if !defined(__CUDA_ARCH__)
			T *__restrict__ ptr_x = std::assume_aligned<mem_align>(x);
			T *__restrict__ ptr_y = std::assume_aligned<mem_align>(y);
			T *__restrict__ ptr_z = std::assume_aligned<mem_align>(z);
			ptr_x[idx] = vec[0]; ptr_y[idx] = vec[1]; ptr_z[idx] = vec[2];
		#else
			x[idx] = vec[0]; y[idx] = vec[1]; z[idx] = vec[2];
		#endif
	}
	VectorFieldView(T *x_new, T *y_new, T *z_new, cuda::std::array<int, 3> num_new, cuda::std::array<T, 3> r_max_new, cuda::std::array<T, 3> inv_dx_new, std::size_t field_size_new)
		: x(x_new), y(y_new), z(z_new), num(num_new), r_max(r_max_new), inv_dx(inv_dx_new), field_size(field_size_new) {}
};

#endif