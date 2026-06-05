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

#include "vtk_output.hpp"
#include "sim_structs.hpp"

template <typename T>
void output_vtk_header_start(FILE *out, const Particles<T> &particles) {
	std::fprintf(out, "# vtk DataFile Version 3.0\n");
	std::fprintf(out, "Volumetric data\n");
	std::fprintf(out, "BINARY\n");
	std::fprintf(out, "DATASET STRUCTURED_POINTS\n");
	std::fprintf(out, "DIMENSIONS %d %d %d\n", particles.num[0], particles.num[1], particles.num[2]);
	std::fprintf(out, "ORIGIN %lf %lf %lf\n", -particles.r_max[0], -particles.r_max[1], -particles.r_max[2]);
	std::fprintf(out, "SPACING %lf %lf %lf\n", T(2.0) * particles.r_max[0] / particles.num[0], T(2.0) * particles.r_max[1] / particles.num[1], T(2.0) * particles.r_max[2] / particles.num[2]);
	std::fprintf(out, "POINT_DATA %d\n", particles.num[0] * particles.num[1] * particles.num[2]);
}

void output_vtk_header_scalar_next(FILE *out, const char *name) {
	std::fprintf(out, "SCALARS %s float 1\n", name);
	std::fprintf(out, "LOOKUP_TABLE default\n");
}

template <typename T>
void output_test(FILE *out, const Particles<T> &particles, const char *name) {
	std::size_t nx = particles.num[0], ny = particles.num[1], nz = particles.num[2], total = nx * ny * nz;
	std::unique_ptr<uint32_t[]> vtk_scalar(new uint32_t[total]);
	#pragma omp parallel for collapse(3)
	for(std::size_t k = 0; k < nz; k++) {
		for(std::size_t j = 0; j < ny; j++) {
			for(std::size_t i = 0; i < nx; i++) {
				int idx = grid_idx(i, j, k, nx, ny, nz);
				int write_idx = (k * ny * nx) + (j * nx) + i;
				vtk_scalar[write_idx] = swap_endian(static_cast<float>(particles.z[idx]));
			}
		}
	}
	output_vtk_header_scalar_next(out, name);
	fwrite(vtk_scalar.get(), sizeof(uint32_t), total, out);
}

template void output_vtk_header_start<double>(FILE *out, const Particles<double> &part);
template void output_test<double>(FILE *out, const Particles<double> &particles, const char *name);

template void output_vtk_header_start<float>(FILE *out, const Particles<float> &part);
template void output_test<float>(FILE *out, const Particles<float> &particles, const char *name);