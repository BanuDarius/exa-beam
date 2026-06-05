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

#include <omp.h>
#include <array>
#include <cstdio>
#include <complex>

#include "sim_structs.hpp"
#include "physics.hpp"

template <typename T>
void start_simulation() {
	Particles<T> particles(128, 128, 32);
	Laser<T> laser(0, 0, T(0.057), T(15.0));
	
	std::array<T, 3> pos = {T(30000.0), T(0.0), T(2.0)};
	std::complex<T> u_new = compute_u(laser, pos);
	std::printf("%0.3lf %0.3lf\n", real(u_new), imag(u_new));
	std::printf("%0.3lf %0.3lf\n", laser.lambda, laser.w0);
}

int main(int argc, char **argv) {
	if(argc != 1) {
		std::fprintf(stderr, "%s BAD ARGUMENTS!", argv[0]);
		return 1;
	}
	double start_time = omp_get_wtime();
	std::printf("Simulation started.\n");
	
	start_simulation<double>();
	
	std::printf("Simulation ended.\n");
	std::printf("Time taken: %0.3lfs.\n", omp_get_wtime() - start_time);
	return 0;
}