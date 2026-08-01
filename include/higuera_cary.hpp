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
__device__ __host__ inline T comp_gamma(cuda::std::array<T, 3> u_vec) noexcept {
	T u2 = dot(u_vec, u_vec);
	T gamma = cuda::std::sqrt(T(1.0) + u2 / (c<T> * c<T>));
	return gamma;
}

template <std::floating_point T>
__device__ __host__ inline T hc_s_factor(cuda::std::array<T, 3> t_rot) noexcept {
	T s_factor = T(2.0) / (T(1.0) + dot(t_rot, t_rot));
	return s_factor;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_beta(cuda::std::array<T, 3> B, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	cuda::std::array<T, 3> beta = B * term;
	return beta;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_epsilon(cuda::std::array<T, 3> E, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	cuda::std::array<T, 3> epsilon = E * term;
	return epsilon;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_t_rot(cuda::std::array<T, 3> beta, T gamma_new) noexcept {
	T term = T(1.0) / gamma_new;
	cuda::std::array<T, 3> t_rot = beta * term;
	return t_rot;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_u_minus(cuda::std::array<T, 3> u, cuda::std::array<T, 3> epsilon) noexcept {
	cuda::std::array<T, 3> u_minus = u + epsilon;
	return u_minus;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_u_prime(cuda::std::array<T, 3> u_minus, cuda::std::array<T, 3> t_rot) noexcept {
	cuda::std::array<T, 3> term = cross(u_minus, t_rot);
	cuda::std::array<T, 3> u_prime = u_minus + term;
	return u_prime;
}

template <std::floating_point T>
__device__ __host__ inline T hc_gamma_new(cuda::std::array<T, 3> u_minus, cuda::std::array<T, 3> beta, T gamma_minus) noexcept {
	T t1 = gamma_minus * gamma_minus - dot(beta, beta);
	T beta_dot_u = dot(beta, u_minus);
	T t2 = dot(beta, beta) + (beta_dot_u * beta_dot_u) / (c<T> * c<T>);
	T t3 = cuda::std::sqrt(T(0.5) * (t1 + cuda::std::sqrt(t1 * t1 + T(4.0) * t2)));
	return t3;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::array<T, 3> hc_u_plus(cuda::std::array<T, 3> u_minus, cuda::std::array<T, 3> u_prime, cuda::std::array<T, 3> t_rot, T s_factor) noexcept {
	t_rot = t_rot * s_factor;
	cuda::std::array<T, 3> term = cross(u_prime, t_rot);
	cuda::std::array<T, 3> u_plus = u_minus + term;
	return u_plus;
}

template <std::floating_point T>
inline void higuera_cary_update(Particles<T> &particles, std::span<const Laser<T>> lasers, T t, T dt) noexcept {
	int laser_count = lasers.front().laser_count;
	ParticlesView<T> particles_view = particles.get_cpu_view();
	#pragma omp parallel for simd schedule(static)
	for(std::size_t idx = 0; idx < particles.particle_num; idx++) {
		T gamma = particles_view.get_gamma(idx);
		cuda::std::array<T, 3> r_vec_global = particles_view.get_position(idx);
		cuda::std::array<T, 3> u_vec_global = particles_view.get_velocity(idx);
		
		T half_dt = T(0.5) * dt, half_dt_gamma = half_dt / gamma;
		cuda::std::array<T, 3> r_half_global = r_vec_global + u_vec_global * half_dt_gamma;
		
		EBVectors<T> eb_vec{};
		for(int i = 0; i < laser_count; i++) {
			cuda::std::array<T, 3> r_half_local = lasers[i].pos_global_to_local(r_half_global);
			EBVectors eb_vec_local = compute_eb(lasers[i], r_half_local, t + half_dt);
			eb_vec = EBVectors(
				eb_vec.e + lasers[i].vec_local_to_global(eb_vec_local.e),
				eb_vec.b + lasers[i].vec_local_to_global(eb_vec_local.b)
			);
		}
		
		cuda::std::array<T, 3> beta = hc_beta(eb_vec.b, dt);
		cuda::std::array<T, 3> epsilon = hc_epsilon(eb_vec.e, dt);
		cuda::std::array<T, 3> u_minus = hc_u_minus(u_vec_global, epsilon);
		
		T gamma_minus = comp_gamma(u_minus);
		T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
		
		cuda::std::array<T, 3> t_rot = hc_t_rot(beta, gamma_new);
		T s_factor = hc_s_factor(t_rot);
		cuda::std::array<T, 3> u_prime = hc_u_prime(u_minus, t_rot);
		cuda::std::array<T, 3> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
		
		cuda::std::array<T, 3> u_final_global = u_plus + epsilon;
		gamma = comp_gamma(u_final_global);
		half_dt_gamma = T(0.5) * dt / gamma;
		r_vec_global = r_half_global + u_final_global * half_dt_gamma;
		
		particles_view.set_gamma(gamma, idx);
		particles_view.set_position(r_vec_global, idx);
		particles_view.set_velocity(u_final_global, idx);
	}
}

template <std::floating_point T> void higuera_cary_update_gpu(Particles<T> &particles, T t, T dt) noexcept;

#endif