// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <cstdio>
#include <cstdlib>
#include <fstream>

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "init.hpp"

template <std::floating_point T>
void read_input_file(const std::string &parameters_filename, Parameters<T> &parameters) {
	T max_dim_mult, tf;
	int i = 0, nx, steps, substeps, laser_count, use_gpu, output_laser_fields;
	
	std::string current;
	std::ifstream input_param(parameters_filename);
	if(!input_param) {
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
		} else if(current == "laser_count") {
			if(input_param >> laser_count) i++;
		} else if(current == "max_dim_mult") {
			if(input_param >> max_dim_mult) i++;
		} else if(current == "output_laser_fields") {
			if(input_param >> output_laser_fields) i++;
		}
	}
	if(i != input_file_count) {
		std::fprintf(stderr, "INVALID INPUT FILE!\n"); std::exit(1);
	}
	parameters = Parameters(nx, steps, substeps, laser_count, tf, max_dim_mult, static_cast<bool>(use_gpu), static_cast<bool>(output_laser_fields));
}

template <std::floating_point T>
void read_lasers_file(const std::string &lasers_filename, std::vector<Laser<T>> &lasers, const Parameters<T> &parameters) {
	cuda::std::array<T, 3> r_0;
	int laser_count = parameters.laser_count, i = 0, j = 0, p, m;
	T a0, tau, psi, phi, theta, omega, w0_mult, zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag;
	
	std::string current;
	std::ifstream input_lasers(lasers_filename);
	if(!input_lasers) {
		std::fprintf(stderr, "CANNOT OPEN LASERS FILE!\n"); std::exit(1);
	}
	while(input_lasers >> current) {
		if(current == "p") {
			if(input_lasers >> p) i++;
		} else if(current == "m") {
			if(input_lasers >> m) i++;
		} else if(current == "a0") {
			if(input_lasers >> a0) i++;
		} else if(current == "tau") {
			if(input_lasers >> tau) i++;
		} else if(current == "phi") {
			if(input_lasers >> phi) i++;
		} else if(current == "psi") {
			if(input_lasers >> psi) i++;
		} else if(current == "theta") {
			if(input_lasers >> theta) i++;
		} else if(current == "omega") {
			if(input_lasers >> omega) i++;
		} else if(current == "w0_mult") {
			if(input_lasers >> w0_mult) i++;
		} else if(current == "r_0_x") {
			if(input_lasers >> r_0[0]) i++;
		} else if(current == "r_0_y") {
			if(input_lasers >> r_0[1]) i++;
		} else if(current == "r_0_z") {
			if(input_lasers >> r_0[2]) i++;
		} else if(current == "zeta_x_real") {
			if(input_lasers >> zeta_x_real) i++;
		} else if(current == "zeta_x_imag") {
			if(input_lasers >> zeta_x_imag) i++;
		} else if(current == "zeta_y_real") {
			if(input_lasers >> zeta_y_real) i++;
		} else if(current == "zeta_y_imag") {
			if(input_lasers >> zeta_y_imag) i++;
		}
		if(i == input_laser_count) {
			i = 0; j++;
			cuda::std::complex<T> zeta_x(zeta_x_real, zeta_x_imag);
			cuda::std::complex<T> zeta_y(zeta_y_real, zeta_y_imag);
			lasers.push_back(Laser(p, m, a0, omega, w0_mult, tau, psi, zeta_x, zeta_y, r_0, phi, theta));
		}
	}
	if(j != laser_count) {
		std::fprintf(stderr, "INVALID LASERS FILE!\n"); std::exit(1);
	}
}

template void read_input_file<double>(const std::string &parameters_filename, Parameters<double> &parameters);
template void read_lasers_file<double>(const std::string &lasers_filename, std::vector<Laser<double>> &lasers, const Parameters<double> &parameters);

template void read_input_file<float>(const std::string &parameters_filename, Parameters<float> &parameters);
template void read_lasers_file<float>(const std::string &lasers_filename, std::vector<Laser<float>> &lasers, const Parameters<float> &parameters);