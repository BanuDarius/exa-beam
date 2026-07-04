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
#include <format>
#include <fstream>
#include <cstdlib>

#include "init.hpp"
#include "physics.hpp"
#include "vtk_output.hpp"
#include "sim_structs.hpp"
#include "higuera_cary.hpp"

template <typename T>
void cpu_simulate(const Parameters<T> &parameters, const Laser<T> &laser, const std::string &output_directory) {
	int steps = parameters.steps, substeps = parameters.substeps, nx = parameters.nx;
	T dt = parameters.tf / steps;
	
	DataVTK data_vtk(nx, nx, nx);
	Particles<T> particles(parameters, laser);
	ComplexScalarField<T> u_field(parameters, laser);
	VectorField<T> e_field(parameters, laser), b_field(parameters, laser);
	
	compute_u_field(u_field, laser);
	for(int step = 0; step < steps; step++) {
		T time = step * dt;
		#pragma omp parallel for schedule(static)
		for(int i = 0; i < nx * nx * nx; i++)
			higuera_cary_step(particles, laser, time, dt, i);
		if(step % substeps == 0) {
			std::string output_filename = std::format("{}/out-{:04d}.vtk", output_directory, step / substeps);
			std::ofstream output_file(output_filename, std::ios::binary);
			if(!output_file) {
				std::fprintf(stderr, "CANNOT OPEN OUTPUT FILE!\n"); std::exit(1);
			}
			compute_eb_field(e_field, b_field, u_field, laser, time);
			
			output_vtk_header(output_file, particles);
			output_vtk_vector_field(output_file, data_vtk, e_field, "E");
			output_vtk_vector_field(output_file, data_vtk, b_field, "B");
			output_vtk_particles_positions(output_file, data_vtk, particles, "pos");
			std::printf("Computed step: %d/%d.\n", step, steps);
		}
	}
}

template <typename T>
void start_simulation(const std::string &input_file, const std::string &output_directory) {
	Laser<T> laser;
	Parameters<T> parameters;
	read_input_file(input_file, parameters, laser);
	cpu_simulate(parameters, laser, output_directory);
}

int main(int argc, char **argv) {
	if(argc != 3) {
		std::fprintf(stderr, "%s BAD ARGUMENTS!\n", argv[0]);
		return 1;
	}
	double start_time = omp_get_wtime();
	std::printf("Simulation started.\n");
	
	start_simulation<double>(argv[1], argv[2]);
	
	std::printf("Simulation ended.\n");
	std::printf("Time taken: %0.3lfs.\n", omp_get_wtime() - start_time);
	return 0;
}