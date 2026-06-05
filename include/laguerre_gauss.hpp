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

template <typename T>
inline T compute_w_z(T w0, T z, T z_r) {
	T w_z = w0 * std::sqrt(T(1.0) + z * z / (z_r * z_r));
	return w_z;
}

template <typename T>
inline T compute_r_z(T z, T z_r) {
	T r_z = z + z_r * z_r / z;
	return r_z;
}

template <typename T>
inline T compute_guoy(T z, T z_r) {
	T psi = std::atan(z / z_r);
	return psi;
}

template <typename T>
inline std::complex<T> compute_u(const Laser<T> &laser, const std::array<T, 3> &r) {
	T w0 = laser.w0; T k = laser.k; T z_r = laser.z_r; T z = r[2];
	T rho2 = r[0] * r[0] + r[1] * r[1];
	
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

#endif