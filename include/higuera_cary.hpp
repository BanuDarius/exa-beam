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

#ifndef HIGUERA_CARY_H
#define HIGUERA_CARY_H

#include <array>
#include <cmath>

#include "sim_structs.hpp"
#include "math_functions.hpp"
#include "laguerre_gauss.hpp"

template <typename T>
inline T comp_gamma(std::array<T, 3> u_vec) {
	T u2 = dot(u_vec, u_vec);
	T gamma = std::sqrt(T(1.0) + u2 / (c<T> * c<T>));
	return gamma;
}

template <typename T>
inline T hc_s_factor(std::array<T, 3> t_rot) noexcept {
	T s_factor = T(2.0) / (T(1.0) + dot(t_rot, t_rot));
	return s_factor;
}

template <typename T>
inline std::array<T, 3> hc_beta(std::array<T, 3> B, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	std::array<T, 3> beta = B * term;
	return beta;
}

template <typename T>
inline std::array<T, 3> hc_epsilon(std::array<T, 3> E, T dt) noexcept {
	T term = e_0<T> * dt / (T(2.0) * m_e<T>);
	std::array<T, 3> epsilon = E * term;
	return epsilon;
}

template <typename T>
inline std::array<T, 3> hc_t_rot(std::array<T, 3> beta, T gamma_new) noexcept {
	T term = T(1.0) / gamma_new;
	std::array<T, 3> t_rot = beta * term;
	return t_rot;
}

template <typename T>
inline std::array<T, 3> hc_u_minus(std::array<T, 3> u, std::array<T, 3> epsilon) noexcept {
	std::array<T, 3> u_minus = u + epsilon;
	return u_minus;
}

template <typename T>
inline std::array<T, 3> hc_u_prime(std::array<T, 3> u_minus, std::array<T, 3> t_rot) noexcept {
	std::array<T, 3> term = cross(u_minus, t_rot);
	std::array<T, 3> u_prime = u_minus + term;
	return u_prime;
}

template <typename T>
inline T hc_gamma_new(std::array<T, 3> u_minus, std::array<T, 3> beta, T gamma_minus) noexcept {
	T t1 = gamma_minus * gamma_minus - dot(beta, beta);
	T beta_dot_u = dot(beta, u_minus);
	T t2 = dot(beta, beta) + (beta_dot_u * beta_dot_u) / (c<T> * c<T>);
	T t3 = std::sqrt(T(0.5) * (t1 + std::sqrt(t1 * t1 + T(4.0) * t2)));
	return t3;
}

template <typename T>
inline std::array<T, 3> hc_u_plus(std::array<T, 3> u_minus, std::array<T, 3> u_prime, std::array<T, 3> t_rot, T s_factor) noexcept {
	t_rot = t_rot * s_factor;
	std::array<T, 3> term = cross(u_prime, t_rot);
	std::array<T, 3> u_plus = u_minus + term;
	return u_plus;
}

template<typename T>
void higuera_cary_step(Particles<T> &particles, const Laser<T> &laser, T t, T dt, int idx) noexcept {
	std::array<T, 3> r_vec = particles.get_position(idx);
	std::array<T, 3> u_vec = particles.get_velocity(idx);
	
	T gamma = particles.gamma[idx];
	T half_dt_gamma = T(0.5) * dt / gamma;
	r_vec += u_vec * half_dt_gamma;
	
	EBVectors eb_vec = compute_eb(laser, r_vec, t + T(0.5) * dt);
	
	std::array<T, 3> beta = hc_beta(eb_vec.b, dt);
	std::array<T, 3> epsilon = hc_epsilon(eb_vec.e, dt);
	std::array<T, 3> u_minus = hc_u_minus(u_vec, epsilon);
	
	T gamma_minus = comp_gamma(u_minus);
	T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
	
	std::array<T, 3> t_rot = hc_t_rot(beta, gamma_new);
	T s_factor = hc_s_factor(t_rot);
	std::array<T, 3> u_prime = hc_u_prime(u_minus, t_rot);
	std::array<T, 3> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
	
	std::array<T, 3> u_final = u_plus + epsilon;
	gamma = comp_gamma(u_final);
	half_dt_gamma = T(0.5) * dt / gamma;
	r_vec += u_final * half_dt_gamma;

	particles.set_position(r_vec, idx);
	particles.set_velocity(u_final, idx);
	particles.gamma[idx] = gamma;
}

#endif