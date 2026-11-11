// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef HIGUERA_CARY_H
#define HIGUERA_CARY_H

#include <span>
#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/cmath>

#include "sim_structs.hpp"
#include "math_functions.hpp"
#include "laguerre_gauss.hpp"

template <std::floating_point T>
__device__ __host__ inline T hc_gamma(Vector3<T> u_vec) noexcept {
	T u2 = dot(u_vec, u_vec);
	return cuda::std::sqrt(T(1.0) + u2 / (c<T> * c<T>));
}

template <std::floating_point T>
__device__ __host__ inline T hc_s_factor(Vector3<T> t_rot) noexcept {
	T s_factor = T(2.0) / (T(1.0) + dot(t_rot, t_rot));
	return s_factor;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_beta(Vector3<T> B, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	Vector3<T> beta = B * term;
	return beta;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_epsilon(Vector3<T> E, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	Vector3<T> epsilon = E * term;
	return epsilon;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_t_rot(Vector3<T> beta, T gamma_new) noexcept {
	T term = T(1.0) / gamma_new;
	return beta * term;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_u_minus(Vector3<T> u, Vector3<T> epsilon) noexcept {
	Vector3<T> u_minus = u + epsilon;
	return u_minus;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_u_prime(Vector3<T> u_minus, Vector3<T> t_rot) noexcept {
	Vector3<T> term = cross(u_minus, t_rot);
	return u_minus + term;
}

template <std::floating_point T>
__device__ __host__ inline T hc_gamma_new(Vector3<T> u_minus, Vector3<T> beta, T gamma_minus) noexcept {
	T t1 = gamma_minus * gamma_minus - dot(beta, beta);
	T beta_dot_u = dot(beta, u_minus);
	T t2 = dot(beta, beta) + (beta_dot_u * beta_dot_u) / (c<T> * c<T>);
	T t3 = cuda::std::sqrt(T(0.5) * (t1 + cuda::std::sqrt(t1 * t1 + T(4.0) * t2)));
	return t3;
}

template <std::floating_point T>
__device__ __host__ inline Vector3<T> hc_u_plus(Vector3<T> u_minus, Vector3<T> u_prime, Vector3<T> t_rot, T s_factor) noexcept {
	t_rot = t_rot * s_factor;
	Vector3<T> term = cross(u_prime, t_rot);
	Vector3<T> u_plus = u_minus + term;
	return u_plus;
}

template <std::floating_point T>
inline void hc_update(Particles<T> &particles, std::span<const Laser<T>> lasers, T t, T dt) noexcept {
	int laser_count = lasers.size();
	ParticlesView<T> particles_view = particles.get_cpu_view();
	#pragma omp parallel for simd schedule(static)
	for(std::size_t idx = 0; idx < particles.particle_num; idx++) {
		Vector3<T> r_vec = particles_view.get_position(idx);
		Vector3<T> u_vec = particles_view.get_velocity(idx);
		Vector3<T> b_vec{}, e_vec{};
		
		Vector3<T> beta = hc_beta(b_vec, dt);
		Vector3<T> epsilon = hc_epsilon(e_vec, dt);
		Vector3<T> u_minus = hc_u_minus(u_vec, epsilon);
		
		T gamma_minus = hc_gamma(u_minus);
		T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
		
		Vector3<T> t_rot = hc_t_rot(beta, gamma_new);
		T s_factor = hc_s_factor(t_rot);
		Vector3<T> u_prime = hc_u_prime(u_minus, t_rot);
		Vector3<T> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
		
		u_vec = u_plus + epsilon;
		T gamma_half = hc_gamma(u_vec);
		r_vec += u_vec * (dt / gamma_half);
		
		particles_view.set_gamma(gamma_half, idx);
		particles_view.set_position(r_vec, idx);
		particles_view.set_velocity(u_vec, idx);
	}
}

template <std::floating_point T> __global__ void hc_update_gpu_kernel(ParticlesView<T> particles_view, __grid_constant__ const DeviceLasers<T> lasers, T t, T dt);
template <std::floating_point T> void hc_update_gpu(Particles<T> &particles, const DeviceLasers<T> &lasers, T t, T dt) noexcept;

#endif