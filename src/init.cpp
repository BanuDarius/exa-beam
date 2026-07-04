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

#include <cstdlib>
#include <cstring>
#include <fstream>

#include "init.hpp"
#include "sim_structs.hpp"

template <typename T>
void read_input_file(const char *input_filename, Parameters<T> &parameters, Laser<T> &laser) {
	double a0, tf, tau, psi, omega, w0_mult, max_dim_mult, zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag;
	int i = 0, p, m, nx, steps, substeps;
	char current[string_size];
	
	FILE *in = fopen(input_filename, "r");
	while(std::fscanf(in, "%s", current) != EOF) {
		if(!std::strcmp(current, "p"))
			i += std::fscanf(in, "%i", &p);
		else if(!std::strcmp(current, "m"))
			i += std::fscanf(in, "%i", &m);
		else if(!std::strcmp(current, "nx"))
			i += std::fscanf(in, "%i", &nx);
		else if(!std::strcmp(current, "a0"))
			i += std::fscanf(in, "%lf", &a0);
		else if(!std::strcmp(current, "tf"))
			i += std::fscanf(in, "%lf", &tf);
		else if(!std::strcmp(current, "tau"))
			i += std::fscanf(in, "%lf", &tau);
		else if(!std::strcmp(current, "psi"))
			i += std::fscanf(in, "%lf", &psi);
		else if(!std::strcmp(current, "omega"))
			i += std::fscanf(in, "%lf", &omega);
		else if(!std::strcmp(current, "steps"))
			i += std::fscanf(in, "%i", &steps);
		else if(!std::strcmp(current, "w0_mult"))
			i += std::fscanf(in, "%lf", &w0_mult);
		else if(!std::strcmp(current, "substeps"))
			i += std::fscanf(in, "%i", &substeps);
		else if(!std::strcmp(current, "zeta_x_real"))
			i += std::fscanf(in, "%lf", &zeta_x_real);
		else if(!std::strcmp(current, "zeta_x_imag"))
			i += std::fscanf(in, "%lf", &zeta_x_imag);
		else if(!std::strcmp(current, "zeta_y_real"))
			i += std::fscanf(in, "%lf", &zeta_y_real);
		else if(!std::strcmp(current, "zeta_y_imag"))
			i += std::fscanf(in, "%lf", &zeta_y_imag);
		else if(!std::strcmp(current, "max_dim_mult"))
			i += std::fscanf(in, "%lf", &max_dim_mult);
	}
	if(i != input_file_count) {
		std::fprintf(stderr, "INVALID INPUT FILE!\n"); std::exit(1);
	}
	std::complex<T> zeta_x = { T(zeta_x_real), T(zeta_x_imag) };
	std::complex<T> zeta_y = { T(zeta_y_real), T(zeta_y_imag) };
	parameters = Parameters(nx, steps, substeps, T(tf), T(max_dim_mult));
	laser = Laser(p, m, T(a0), T(omega), T(w0_mult), T(tau), T(psi), zeta_x, zeta_y);
	fclose(in);
}

template void read_input_file<double>(const char *input_filename, Parameters<double> &parameters, Laser<double> &laser);

template void read_input_file<float>(const char *input_filename, Parameters<float> &parameters, Laser<float> &laser);