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
#include <cstring>
#include <concepts>

#include "init.hpp"
#include "simulation.hpp"
#include "sim_structs.hpp"

template <std::floating_point T>
void start_simulation(const std::string &input_file, const std::string &output_directory) {
	Laser<T> laser;
	Parameters<T> parameters;
	read_input_file(input_file, parameters, laser);
	cpu_simulate(parameters, laser, output_directory);
}

int main(int argc, char **argv) {
	if(argc > 4) {
		std::fprintf(stderr, "%s BAD ARGUMENTS!\n", argv[0]);
		return 1;
	}
	double start_time = omp_get_wtime();
	std::printf("Simulation started.\n");
	
	if(!strcmp(argv[1], "--float"))
		start_simulation<float>(argv[2], argv[3]);
	else
		start_simulation<double>(argv[1], argv[2]);
	
	std::printf("Simulation ended.\n");
	std::printf("Time taken: %0.3lfs.\n", omp_get_wtime() - start_time);
	return 0;
}