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
	int steps = parameters.steps, substeps = parameters.substeps;
	cuda::std::array<float, 2> diagnostics;
	bool use_gpu = parameters.use_gpu;
	T dt = parameters.tf / steps;
	
	std::string filename_diagnostics = std::format("{}/diagnostics.bin", output_directory);
	std::ofstream output_diagnostics(filename_diagnostics, std::ios::binary);
	if(!output_diagnostics) { std::fprintf(stderr, "Cannot open output diagnostics file!\n"); return; }
	
	DataVTK data_vtk(parameters);
	DeviceLasers<T> device_lasers(lasers);
	Particles<T> particles(parameters, lasers.front());
	ScalarField<T> lz_field(parameters, lasers.front());
	VectorField<T> e_field(parameters, lasers.front()), b_field(parameters, lasers.front());
	
	if(use_gpu) initialize_eb_fields_gpu(e_field, b_field, device_lasers, dt);
	else initialize_eb_fields(e_field, b_field, lasers, dt);
	
	for(int step = 0; step < steps; step++) {
		//T t = step * dt;
		//if(use_gpu) hc_update_gpu(particles, device_lasers, t, dt);
		//else hc_update(particles, lasers, t, dt);
		
		if(use_gpu) {
			fdtd_update_b_gpu(b_field, e_field, dt);
			fdtd_update_e_gpu(e_field, b_field, dt);
		} else {
			fdtd_update_b(b_field, e_field, dt);
			fdtd_update_e(e_field, b_field, dt);
		}
		
		if(step % substeps == 0) {
			if(use_gpu) {
				e_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
				b_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
				particles.transfer_data_gpu_to_cpu(cudaStreamDefault);
			}
			if(use_gpu) diagnostics[0] = total_energy_gpu(e_field, b_field);
			else diagnostics[0] = total_energy(e_field, b_field);
			
			if(use_gpu) diagnostics[1] = divergence_e_gpu(e_field);
			else diagnostics[1] = divergence_e(e_field);
			
			output_diagnostics.write(reinterpret_cast<const char*>(diagnostics.data()), sizeof(float) * diagnostics.size());
			
			std::string filename_fields = std::format("{}/fields-{:04d}.vtk", output_directory, step / substeps);
			std::ofstream output_fields(filename_fields, std::ios::binary);
			if(!output_fields) { std::fprintf(stderr, "Cannot open output fields file!\n"); return; }
			
			std::string filename_particles = std::format("{}/particles-{:04d}.vtk", output_directory, step / substeps);
			std::ofstream output_particles(filename_particles, std::ios::binary);
			if(!output_particles) { std::fprintf(stderr, "Cannot open output particles file!\n"); return; }
			
			cudaDeviceSynchronize();
			output_vtk_header(output_fields, e_field);
			output_vtk_particles(output_particles, data_vtk, particles);
			output_vtk_vector_field(output_fields, data_vtk, e_field, "E");
			output_vtk_vector_field(output_fields, data_vtk, b_field, "B");
			std::printf("Computed step: %d/%d.\n", step, steps);
		}
	}
	if(use_gpu) {
		compute_lz_gpu(lz_field, particles);
		lz_field.transfer_data_gpu_to_cpu(cudaStreamDefault);
	} else compute_lz(lz_field, particles);
	
	std::string filename_lz = std::format("{}/lz.vtk", output_directory);
	std::ofstream output_lz(filename_lz, std::ios::binary);
	if(!output_lz) {
		std::fprintf(stderr, "Cannot open output Lz file!\n"); return;
	}
	
	cudaDeviceSynchronize();
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
	
	simulate<T>(parameters, lasers, output_directory);
}

int main(int argc, char **argv) {
	if(argc > 5) {
		std::fprintf(stderr, "%s Invalid arguments!\n", argv[0]);
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