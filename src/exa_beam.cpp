// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include <span>
#include <omp.h>
#include <vector>
#include <format>
#include <cstdio>
#include <cstring>
#include <concepts>

#include "init.hpp"
#include "physics.hpp"
#include "vtk_output.hpp"
#include "physics_gpu.hpp"
#include "sim_structs.hpp"
#include "higuera_cary.hpp"

template <std::floating_point T>
void simulate(const Parameters<T> &parameters, std::span<const Laser<T>> lasers, const std::string &output_directory) {
	bool use_gpu = parameters.use_gpu, output_laser_fields = parameters.output_laser_fields;
	int steps = parameters.steps, substeps = parameters.substeps;
	T dt = parameters.tf / steps;
	
	DataVTK data_vtk(parameters);
	Particles<T> particles(parameters, lasers.front());
	ScalarField<T> lz_field(parameters, lasers.front());
	ComplexScalarField<T> u_field(parameters, lasers.front());
	VectorField<T> e_field(parameters, lasers.front()), b_field(parameters, lasers.front());
	
	if(output_laser_fields) {
		if(use_gpu) {
			compute_u_field_gpu(u_field);
			u_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
			cudaDeviceSynchronize();
		} else compute_u_field(u_field, lasers);
	}
	for(int step = 0; step < steps; step++) {
		T t = step * dt;
		if(use_gpu) higuera_cary_update_gpu(particles, t, dt);
		else higuera_cary_update(particles, lasers, t, dt);
		if(step % substeps == 0) {
			if(use_gpu) particles.transfer_data_gpu_to_cpu(cudaStreamDefault);
			if(output_laser_fields) {
				if(use_gpu) {
					compute_eb_field_gpu(e_field, b_field, t);
					e_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
					b_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
				} else compute_eb_field(e_field, b_field, lasers, t);
				
				std::string filename_fields = std::format("{}/fields-{:04d}.vtk", output_directory, step / substeps);
				std::ofstream output_fields(filename_fields, std::ios::binary);
				
				if(!output_fields) {
					std::fprintf(stderr, "CANNOT OPEN OUTPUT FIELD FILE!\n"); return;
				}
				
				cudaDeviceSynchronize();
				output_vtk_header(output_fields, e_field);
				output_vtk_vector_field(output_fields, data_vtk, e_field, "E");
				output_vtk_vector_field(output_fields, data_vtk, b_field, "B");
				output_vtk_complex_scalar_field(output_fields, data_vtk, u_field, "u00");
			}
			
			std::string filename_particles = std::format("{}/particles-{:04d}.vtk", output_directory, step / substeps);
			std::ofstream output_particles(filename_particles, std::ios::binary);
			
			if(!output_particles) {
				std::fprintf(stderr, "CANNOT OPEN OUTPUT PARTICLES FILE!\n"); return;
			}
			
			cudaDeviceSynchronize();
			output_vtk_particles(output_particles, data_vtk, particles);
			std::printf("Computed step: %d/%d.\n", step, steps);
		}
	}
	if(use_gpu) {
		compute_lz_gpu(lz_field, particles);
		lz_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
		cudaDeviceSynchronize();
	} else compute_lz(lz_field, particles);
	
	std::string filename_lz = std::format("{}/lz.vtk", output_directory);
	std::ofstream output_lz(filename_lz, std::ios::binary);
	if(!output_lz) {
		std::fprintf(stderr, "CANNOT OPEN OUTPUT LZ FILE!\n"); return;
	}
	
	output_vtk_header(output_lz, lz_field);
	output_vtk_scalar_field(output_lz, data_vtk, lz_field, "Lz");
}

template <std::floating_point T>
void start_simulation(const std::string &input_file, const std::string &input_laser, const std::string &output_directory) {
	Parameters<T> parameters;
	read_input_file(input_file, parameters);
	
	std::vector<Laser<T>> lasers;
	lasers.reserve(parameters.laser_count);
	read_lasers_file(input_laser, lasers, parameters);
	if(parameters.use_gpu) transfer_lasers_cpu_to_gpu<T>(lasers);
	
	simulate<T>(parameters, lasers, output_directory);
}

int main(int argc, char **argv) {
	if(argc > 5) {
		std::fprintf(stderr, "%s BAD ARGUMENTS!\n", argv[0]);
		return 1;
	}
	double start_time = omp_get_wtime();
	std::printf("Simulation started.\n");
	
	if(!strcmp(argv[1], "--float"))
		start_simulation<float>(argv[2], argv[3], argv[4]);
	else
		start_simulation<double>(argv[1], argv[2], argv[3]);
	
	std::printf("Simulation ended.\n");
	std::printf("Time taken: %0.3lfs.\n", omp_get_wtime() - start_time);
	return 0;
}