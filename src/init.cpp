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

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <complex>

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "init.hpp"

template <std::floating_point T>
void read_input_file(const std::string &input_filename, Parameters<T> &parameters, Laser<T> &laser) {
	double a0, tf, tau, psi, omega, w0_mult, max_dim_mult, zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag;
	int i = 0, p, m, nx, steps, substeps, use_gpu;
	
	std::string current;
	std::ifstream in(input_filename);
	if(!in) {
		std::fprintf(stderr, "CANNOT OPEN INPUT FILE!\n"); std::exit(1);
	}
	while(in >> current) {
		if(current == "p") {
			if(in >> p) i++;
		} else if(current == "m") {
			if(in >> m) i++;
		} else if(current == "nx") {
			if(in >> nx) i++;
		} else if(current == "a0") {
			if(in >> a0) i++;
		} else if(current == "tf") {
			if(in >> tf) i++;
		} else if(current == "tau") {
			if(in >> tau) i++;
		} else if(current == "psi") {
			if(in >> psi) i++;
		} else if(current == "omega") {
			if(in >> omega) i++;
		} else if(current == "steps") {
			if(in >> steps) i++;
		} else if(current == "use_gpu") {
			if(in >> use_gpu) i++;
		} else if(current == "w0_mult") {
			if(in >> w0_mult) i++;
		} else if(current == "substeps") {
			if(in >> substeps) i++;
		} else if(current == "zeta_x_real") {
			if(in >> zeta_x_real) i++;
		} else if(current == "zeta_x_imag") {
			if(in >> zeta_x_imag) i++;
		} else if(current == "zeta_y_real") {
			if(in >> zeta_y_real) i++;
		} else if(current == "zeta_y_imag") {
			if(in >> zeta_y_imag) i++;
		} else if(current == "max_dim_mult") {
			if(in >> max_dim_mult) i++;
		}
	}
	if(i != input_file_count) {
		std::fprintf(stderr, "INVALID INPUT FILE!\n"); std::exit(1);
	}
	cuda::std::complex<T> zeta_x = { T(zeta_x_real), T(zeta_x_imag) };
	cuda::std::complex<T> zeta_y = { T(zeta_y_real), T(zeta_y_imag) };
	parameters = Parameters(nx, steps, substeps, T(tf), T(max_dim_mult), static_cast<bool>(use_gpu));
	laser = Laser(p, m, T(a0), T(omega), T(w0_mult), T(tau), T(psi), zeta_x, zeta_y);
}

template void read_input_file<double>(const std::string &input_filename, Parameters<double> &parameters, Laser<double> &laser);

template void read_input_file<float>(const std::string &input_filename, Parameters<float> &parameters, Laser<float> &laser);