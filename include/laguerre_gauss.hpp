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

#ifndef LAGUERRE_GAUSS_H
#define LAGUERRE_GAUSS_H

#include <cmath>
#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/complex>

#include "sim_structs.hpp"
#include "math_functions.hpp"

template <std::floating_point T>
__device__ __host__ inline T env(T chi, T tau) noexcept {
	using std::exp;
	T term = T(4.0) * pi<T> * pi<T> * tau * tau;
	T x = exp(-(chi * chi) / term);
	return x;
}

template <std::floating_point T>
__device__ __host__ inline T compute_w_z(T w0, T z, T z_r) noexcept {
	using std::sqrt;
	T w_z = w0 * sqrt(T(1.0) + z * z / (z_r * z_r));
	return w_z;
}

template <std::floating_point T>
__device__ __host__ inline T compute_r_z(T z, T z_r) noexcept {
	T r_z = z + z_r * z_r / z;
	return r_z;
}

template <std::floating_point T>
__device__ __host__ inline T compute_guoy(T z, T z_r) noexcept {
	using std::atan;
	T psi = atan(z / z_r);
	return psi;
}

template <std::floating_point T>
__device__ __host__ inline T compute_phi(T x, T y) noexcept {
	using std::atan2;
	T phi = atan2(y, x);
	return phi;
}

template <std::floating_point T>
__device__ __host__ inline cuda::std::complex<T> compute_u(const Laser<T> &laser, cuda::std::array<T, 3> r_vec, T r_z, T w_z) noexcept {
	using std::exp; using std::cos; using std::sin; using std::abs;
	T w0 = laser.w0, k = laser.k, z_r = laser.z_r;
	T x = r_vec[0], y = r_vec[1], z = r_vec[2];
	T rho = sqrt(x * x + y * y);
	T w_z2 = w_z * w_z, rho2 = rho * rho;
	int m = laser.m;
	
	T psi_g = compute_guoy(z, z_r);
	T amplitude, phase, phi;
	
	if(m == 0) {
		amplitude = w0 / w_z * exp(-rho2 / w_z2);
		phase = - k * rho2 / (T(2.0) * r_z) + psi_g;
	} else if(abs(m) == 1) {
		phi = compute_phi(x, y);
		amplitude = sqrt(T(2.0)) * w0 * rho / w_z2 * exp(- rho2 / w_z2);
		phase = -m * phi - k * rho2 / (T(2.0) * r_z) + T(2.0) * psi_g;
	}
	
	T real = amplitude * cos(phase);
	T imag = amplitude * sin(phase);
	cuda::std::complex<T> u(real, imag);
	return u;
}

template <std::floating_point T>
__device__ __host__ inline EBVectors<T> compute_eb(const Laser<T> &laser, cuda::std::array<T, 3> r_vec, T t) noexcept {
	using std::cos; using std::sin; using std::abs;
	T w0 = laser.w0, k = laser.k, z_r = laser.z_r, E0 = laser.E0, tau = laser.tau, psi = laser.psi;
	cuda::std::complex<T> zeta_x = laser.zeta_x, zeta_y = laser.zeta_y;
	T x = r_vec[0], y = r_vec[1], z = r_vec[2];
	int m = laser.m;
	
	T r_z = compute_r_z(z, z_r);
	T w_z = compute_w_z(w0, z, z_r);
	T chi = laser.omega * t - k * z + psi;
	
	cuda::std::complex<T> u_pm = compute_u(laser, r_vec, r_z, w_z);
	cuda::std::complex<T> phase(cos(chi), sin(chi));
	u_pm *= E0 * phase * env(chi, tau);
	
	cuda::std::complex<T> field_term(T(1.0) / r_z, -T(2.0) / (k * w_z * w_z));
	
	cuda::std::complex<T> e_z = field_term * (zeta_x * x + zeta_y * y);
	cuda::std::complex<T> b_z = field_term * (zeta_x * y - zeta_y * x);
	
	if(abs(m) == 1) {
		T rho2 = x * x + y * y;
		T ampl = T(1.0) / (k * rho2);
		cuda::std::complex<T> I(T(0.0), T(1.0));
		
		T m_t = static_cast<T>(m);
		cuda::std::complex<T> e_term = ampl * (m_t * (x * zeta_y - y * zeta_x) + I * (x * zeta_x + y * zeta_y));
		cuda::std::complex<T> b_term = ampl * (m_t * (x * zeta_x + y * zeta_y) + I * (y * zeta_x - x * zeta_y));
		
		e_z += e_term;
		b_z += b_term;
	}
	
	cuda::std::array<T, 3> e_vec = {
		cuda::std::real(u_pm * zeta_x),
		cuda::std::real(u_pm * zeta_y),
		cuda::std::real(u_pm * e_z)
	};
	cuda::std::array<T, 3> b_vec = {
		-cuda::std::real(u_pm * zeta_y) / c<T>,
		cuda::std::real(u_pm * zeta_x) / c<T>,
		cuda::std::real(u_pm * b_z) / c<T>
	};
	EBVectors eb_vec(e_vec, b_vec);
	return eb_vec;
}

template <std::floating_point T>
__device__ __host__ inline EBVectors<T> compute_eb(ComplexScalarFieldView<T> &u_field_view, const Laser<T> &laser, cuda::std::array<T, 3> r_vec, T t, std::size_t idx) noexcept {
	using std::cos; using std::sin; using std::abs;
	T w0 = laser.w0, k = laser.k, z_r = laser.z_r, E0 = laser.E0, tau = laser.tau, psi = laser.psi;
	cuda::std::complex<T> zeta_x = laser.zeta_x, zeta_y = laser.zeta_y;
	T x = r_vec[0], y = r_vec[1], z = r_vec[2];
	int m = laser.m;
	
	T r_z = compute_r_z(z, z_r);
	T w_z = compute_w_z(w0, z, z_r);
	T chi = laser.omega * t - k * z + psi;
	
	cuda::std::complex<T> u_pm = u_field_view.get_field(idx);
	cuda::std::complex<T> phase(cos(chi), sin(chi));
	u_pm *= E0 * phase * env(chi, tau);
	
	cuda::std::complex<T> field_term(T(1.0) / r_z, -T(2.0) / (k * w_z * w_z));
	
	cuda::std::complex<T> e_z = field_term * (zeta_x * x + zeta_y * y);
	cuda::std::complex<T> b_z = field_term * (zeta_x * y - zeta_y * x);
	
	if(abs(m) == 1) {
		T rho2 = x * x + y * y;
		T ampl = T(1.0) / (k * rho2);
		cuda::std::complex<T> I(T(0.0), T(1.0));
		
		T m_t = static_cast<T>(m);
		cuda::std::complex<T> e_term = ampl * (m_t * (x * zeta_y - y * zeta_x) + I * (x * zeta_x + y * zeta_y));
		cuda::std::complex<T> b_term = ampl * (m_t * (x * zeta_x + y * zeta_y) + I * (y * zeta_x - x * zeta_y));
		
		e_z += e_term;
		b_z += b_term;
	}
	
	cuda::std::array<T, 3> e_vec = {
		cuda::std::real(u_pm * zeta_x),
		cuda::std::real(u_pm * zeta_y),
		cuda::std::real(u_pm * e_z)
	};
	cuda::std::array<T, 3> b_vec = {
		-cuda::std::real(u_pm * zeta_y) / c<T>,
		cuda::std::real(u_pm * zeta_x) / c<T>,
		cuda::std::real(u_pm * b_z) / c<T>
	};
	EBVectors eb_vec(e_vec, b_vec);
	return eb_vec;
}

#endif