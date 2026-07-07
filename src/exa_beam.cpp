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
#include <format>
#include <cstdio>
#include <cstring>
#include <concepts>

#include "init.hpp"
#include "physics.hpp"
#include "vtk_output.hpp"
#include "sim_structs.hpp"
#include "higuera_cary.hpp"

#include "physics_gpu.hpp"

template <std::floating_point T>
void simulate(const Parameters<T> &parameters, const Laser<T> &laser, const std::string &output_directory) {
	int steps = parameters.steps, substeps = parameters.substeps;
	T dt = parameters.tf / steps;
	bool use_gpu = parameters.use_gpu;
	
	DataVTK data_vtk(parameters);
	Particles<T> particles(parameters, laser);
	ScalarField<T> lz_field(parameters, laser);
	ComplexScalarField<T> u_field(parameters, laser);
	VectorField<T> e_field(parameters, laser), b_field(parameters, laser);
	
	if(use_gpu) compute_u_field_gpu(u_field, laser);
	else compute_u_field(u_field, laser);
	for(int step = 0; step < steps; step++) {
		T time = step * dt;
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < particles.particle_num; i++)
			higuera_cary_step(particles, laser, time, dt, i);
		if(step % substeps == 0) {
			std::string filename_fields = std::format("{}/out-fields-{:04d}.vtk", output_directory, step / substeps);
			std::string filename_particles = std::format("{}/out-particles-{:04d}.vtk", output_directory, step / substeps);
			
			std::ofstream output_fields(filename_fields, std::ios::binary);
			std::ofstream output_particles(filename_particles, std::ios::binary);
			if(!output_fields || !output_particles) {
				std::fprintf(stderr, "CANNOT OPEN OUTPUT FILES!\n"); return;
			}
			if(use_gpu) compute_eb_field_gpu(e_field, b_field, u_field, laser, time);
			else compute_eb_field(e_field, b_field, u_field, laser, time);
			
			output_vtk_header(output_fields, e_field);
			output_vtk_vector_field(output_fields, data_vtk, e_field, "E");
			output_vtk_vector_field(output_fields, data_vtk, b_field, "B");
			output_vtk_complex_scalar_field(output_fields, data_vtk, u_field, "u00");
			
			output_vtk_particles(output_particles, data_vtk, particles);
			std::printf("Computed step: %d/%d.\n", step, steps);
		}
	}
	std::string filename_lz = std::format("{}/out-lz.vtk", output_directory);
	std::ofstream output_lz(filename_lz, std::ios::binary);
	if(!output_lz) {
		std::fprintf(stderr, "CANNOT OPEN OUTPUT LZ FILE!\n"); return;
	}
	if(use_gpu) compute_lz_gpu(lz_field, particles);
	else compute_lz(lz_field, particles);
	
	output_vtk_header(output_lz, lz_field);
	output_vtk_scalar_field(output_lz, data_vtk, lz_field, "Lz");
}

template <std::floating_point T>
void start_simulation(const std::string &input_file, const std::string &output_directory) {
	Laser<T> laser;
	Parameters<T> parameters;
	read_input_file(input_file, parameters, laser);
	simulate(parameters, laser, output_directory);
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