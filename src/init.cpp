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

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "init.hpp"

template <std::floating_point T>
void read_input_file(const std::string &parameters_filename, const std::string &laser_filename, Parameters<T> &parameters, Laser<T> &laser) {
	cuda::std::array<T, 3> r_0;
	T a0, tf, tau, psi, omega, w0_mult, max_dim_mult, phi, theta, zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag;
	int i = 0, p, m, nx, steps, substeps, use_gpu, output_laser_fields;
	
	std::string current;
	std::ifstream input_param(parameters_filename), input_laser(laser_filename);
	if(!input_param || !input_laser) {
		std::fprintf(stderr, "CANNOT OPEN INPUT FILE!\n"); std::exit(1);
	}
	while(input_param >> current) {
		if(current == "nx") {
			if(input_param >> nx) i++;
		} else if(current == "tf") {
			if(input_param >> tf) i++;
		} else if(current == "steps") {
			if(input_param >> steps) i++;
		} else if(current == "use_gpu") {
			if(input_param >> use_gpu) i++;
		} else if(current == "substeps") {
			if(input_param >> substeps) i++;
		} else if(current == "max_dim_mult") {
			if(input_param >> max_dim_mult) i++;
		} else if(current == "output_laser_fields") {
			if(input_param >> output_laser_fields) i++;
		}
	}
	if(i != input_file_count) {
		std::fprintf(stderr, "INVALID INPUT FILE!\n"); std::exit(1);
	}
	i = 0;
	while(input_laser >> current) {
		if(current == "p") {
			if(input_laser >> p) i++;
		} else if(current == "m") {
			if(input_laser >> m) i++;
		} else if(current == "a0") {
			if(input_laser >> a0) i++;
		} else if(current == "tau") {
			if(input_laser >> tau) i++;
		} else if(current == "psi") {
			if(input_laser >> psi) i++;
		} else if(current == "omega") {
			if(input_laser >> omega) i++;
		} else if(current == "w0_mult") {
			if(input_laser >> w0_mult) i++;
		} else if(current == "zeta_x_real") {
			if(input_laser >> zeta_x_real) i++;
		} else if(current == "zeta_x_imag") {
			if(input_laser >> zeta_x_imag) i++;
		} else if(current == "zeta_y_real") {
			if(input_laser >> zeta_y_real) i++;
		} else if(current == "zeta_y_imag") {
			if(input_laser >> zeta_y_imag) i++;
		} else if(current == "theta") {
			if(input_laser >> theta) i++;
		} else if(current == "phi") {
			if(input_laser >> phi) i++;
		} else if(current == "r0_x") {
			if(input_laser >> r_0[0]) i++;
		} else if(current == "r0_y") {
			if(input_laser >> r_0[1]) i++;
		} else if(current == "r0_z") {
			if(input_laser >> r_0[2]) i++;
		}
	}
	if(i != input_laser_count) {
		std::fprintf(stderr, "INVALID LASER FILE!\n"); std::exit(1);
	}
	cuda::std::complex<T> zeta_x = { zeta_x_real, zeta_x_imag };
	cuda::std::complex<T> zeta_y = { zeta_y_real, zeta_y_imag };
	parameters = Parameters(nx, steps, substeps, tf, max_dim_mult, static_cast<bool>(use_gpu), static_cast<bool>(output_laser_fields));
	laser = Laser(p, m, a0, omega, w0_mult, tau, psi, zeta_x, zeta_y, phi, theta, r_0);
}

template void read_input_file<double>(const std::string &parameters_filename, const std::string &laser_filename, Parameters<double> &parameters, Laser<double> &laser);

template void read_input_file<float>(const std::string &parameters_filename, const std::string &laser_filename, Parameters<float> &parameters, Laser<float> &laser);