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
#include "laguerre_gauss.hpp"

template <typename T>
void test_u(Particles<T> &particles, const Laser<T> &laser) {
	std::size_t total = particles.num[0] * particles.num[1] * particles.num[2];
	#pragma omp parallel for schedule(static)
	for(std::size_t i = 0; i < total; i++) {
		std::array<T, 3> r_i= { particles.x[i], particles.y[i], particles.z[i] };
		std::complex<T> u_i = compute_u(laser, r_i);
		particles.z[i] = real(u_i);
	}
}

template void test_u<double>(Particles<double> &particles, const Laser<double> &laser);

template void test_u<float>(Particles<float> &particles, const Laser<float> &laser);