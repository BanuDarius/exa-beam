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
#include <cstdio>
#include <cstdlib>

#include "physics.hpp"
#include "vtk_output.hpp"
#include "sim_structs.hpp"

template <typename T>
void start_simulation(const char *output_directory) {
	Laser<T> laser(0, 0, T(0.057), T(15.0));
	Particles<T> particles(32, 32, 32, laser.w0);
	ScalarField<T> test_field_u(32, 32, 32, laser.w0);
	test_u(test_field_u, laser);
	
	char output_filename[string_size];
	std::sprintf(output_filename, "%s/out.vtk", output_directory);
	std::FILE *output_file = std::fopen(output_filename, "wb");
	if(output_file == nullptr) {
		std::fprintf(stderr, "CANNOT OPEN OUTPUT FILE!\n"); std::exit(1);
	}
	
	output_vtk_header_start(output_file, test_field_u);
	output_test(output_file, test_field_u, "u00");
	std::fclose(output_file);
}

int main(int argc, char **argv) {
	if(argc != 2) {
		std::fprintf(stderr, "%s BAD ARGUMENTS!\n", argv[0]);
		return 1;
	}
	double start_time = omp_get_wtime();
	std::printf("Simulation started.\n");
	
	start_simulation<double>(argv[1]);
	
	std::printf("Simulation ended.\n");
	std::printf("Time taken: %0.3lfs.\n", omp_get_wtime() - start_time);
	return 0;
}