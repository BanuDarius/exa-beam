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
#include <complex>

#include "sim_structs.hpp"

template <typename T>
inline T env(T chi, T tau) noexcept {
	T term = T(4.0) * pi<T> * pi<T> * tau * tau; 
	T x = std::exp(-(chi * chi) / term);
	return x;
}

template <typename T>
inline T compute_w_z(T w0, T z, T z_r) noexcept {
	T w_z = w0 * std::sqrt(T(1.0) + z * z / (z_r * z_r));
	return w_z;
}

template <typename T>
inline T compute_r_z(T z, T z_r) noexcept {
	T r_z = z + z_r * z_r / z;
	return r_z;
}

template <typename T>
inline T compute_guoy(T z, T z_r) noexcept {
	T psi = std::atan(z / z_r);
	return psi;
}

template <typename T>
inline std::complex<T> compute_u(const Laser<T> &laser, const std::array<T, 3> &r_vec) noexcept {
	T w0 = laser.w0; T k = laser.k; T z_r = laser.z_r; T z = r_vec[2];
	T rho2 = r_vec[0] * r_vec[0] + r_vec[1] * r_vec[1];
	
	T r_z = compute_r_z(z, z_r);
	T psi_g = compute_guoy(z, z_r);
	T w_z = compute_w_z(w0, z, z_r);
	
	T amplitude = w0 / w_z * std::exp(-rho2 / (w_z * w_z));
	T phase = -k * rho2 / (T(2.0) * r_z) + psi_g;
	
	T real = amplitude * std::cos(phase);
	T imag = amplitude * std::sin(phase);
	std::complex<T> u(real, imag);
	return u;
}

template <typename T>
inline std::array<T, 3> compute_e(const ComplexScalarField<T> &u_field, const Laser<T> &laser, const std::array<T, 3> &r_vec, T t, int idx) noexcept {
	T k = laser.k, tau = laser.tau;
	std::complex<T> zeta_x = laser.zeta_x, zeta_y = laser.zeta_y;
	
	T chi = laser.omega * t - k * r_vec[2];
	std::complex<T> u_pm = u_field.v[idx];
	std::complex<T> phase(std::cos(chi), std::sin(chi));
	
	u_pm *= phase * env(chi, tau);
	std::array<T, 3> e_vec = {
		std::real(u_pm * zeta_x),
		std::real(u_pm * zeta_y),
		T(0.0)
	};
	return e_vec;
}

#endif