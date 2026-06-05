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

#include "physics.hpp"

template <typename T>
std::complex<T> compute_u(const Laser<T> &laser, const std::array<T, 3> &pos) {
	T w0 = laser.w0; T k = laser.k; T z_r = laser.z_r; T z = pos[2];
	T rho2 = pos[0] * pos[0] + pos[1] * pos[1];
	
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

template std::complex<double> compute_u<double>(const Laser<double> &laser, const std::array<double, 3> &pos);

template std::complex<float> compute_u<float>(const Laser<float> &laser, const std::array<float, 3> &pos);