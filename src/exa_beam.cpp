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
#include <fstream>
#include <cstdlib>

#include "physics.hpp"
#include "vtk_output.hpp"
#include "sim_structs.hpp"
#include "higuera_cary.hpp"

template <typename T>
void start_simulation(const char *output_directory) {
	std::complex<T> zeta_x(T(0.707), T(0.0));
	std::complex<T> zeta_y(T(0.0), -T(0.707));
	
	T dt = T(5.0), tau = T(8.0);
	int max_steps = 5000, substeps = 10, nx = 32, total = nx * nx * nx;
	Laser<T> laser(0, 0, T(15.0), T(0.057), T(32.0), tau, -T(32.0) * tau, zeta_x, zeta_y);
	VectorField<T> e_field(nx, nx, nx, laser.w0), b_field(nx, nx, nx, laser.w0);
	ComplexScalarField<T> u_field(nx, nx, nx, laser.w0);
	compute_u_field(u_field, laser);
	Particles<T> particles(nx, nx, nx, laser.w0);
	
	for(int step = 0; step < max_steps; step++) {
		T time = step * dt;
		compute_eb_field(e_field, b_field, u_field, laser, time);
		#pragma omp parallel for schedule(static)
		for(int i = 0; i < total; i++)
			higuera_cary_step(particles, laser, time, dt, i);
		if(step % substeps == 0) {
			char output_filename[string_size];
			std::sprintf(output_filename, "%s/out-%04d.vtk", output_directory, step / substeps);
			std::ofstream output_file(output_filename, std::ios::binary);
			if(!output_file) {
				std::fprintf(stderr, "CANNOT OPEN OUTPUT FILE!\n"); std::exit(1);
			}
			output_vtk_header(output_file, particles);
			output_vtk_particles_positions(output_file, particles, "pos");
			output_vtk_vector_field(output_file, e_field, "E");
			output_vtk_vector_field(output_file, b_field, "B");
			std::printf("Computed step: %d/%d.\n", step, max_steps);
		}
	}
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