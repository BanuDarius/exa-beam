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

#include "vtk_output.hpp"
#include "sim_structs.hpp"

template <typename T>
void output_vtk_header(std::ofstream &output_file, const ScalarField<T> &field) {
	int nx = field.num[0], ny = field.num[1], nz = field.num[2];
	T r_max_x = field.r_max[0], r_max_y = field.r_max[1], r_max_z = field.r_max[2];
	output_file << "# vtk DataFile Version 3.0\n";
	output_file << "Volumetric data\n";
	output_file << "BINARY\n";
	output_file << "DATASET STRUCTURED_POINTS\n";
	output_file << "DIMENSIONS " << nx << " " << ny << " "  << nz << "\n";
	output_file << "ORIGIN " << -r_max_x << " " << -r_max_y << " " <<  -r_max_z << "\n";
	output_file << "SPACING " << T(2.0) * r_max_x / nx << " " << T(2.0) * r_max_y / ny << " " << T(2.0) * r_max_z / nz << "\n";
	output_file << "POINT_DATA " << nx * ny * nz << "\n";
}

template <typename T>
void output_vtk_header(std::ofstream &output_file, const VectorField<T> &field) {
	int nx = field.num[0], ny = field.num[1], nz = field.num[2];
	T r_max_x = field.r_max[0], r_max_y = field.r_max[1], r_max_z = field.r_max[2];
	output_file << "# vtk DataFile Version 3.0\n";
	output_file << "Volumetric data\n";
	output_file << "BINARY\n";
	output_file << "DATASET STRUCTURED_POINTS\n";
	output_file << "DIMENSIONS " << nx << " " << ny << " "  << nz << "\n";
	output_file << "ORIGIN " << -r_max_x << " " << -r_max_y << " " <<  -r_max_z << "\n";
	output_file << "SPACING " << T(2.0) * r_max_x / nx << " " << T(2.0) * r_max_y / ny << " " << T(2.0) * r_max_z / nz << "\n";
	output_file << "POINT_DATA " << nx * ny * nz << "\n";
}

void output_vtk_scalar_next(std::ofstream &output_file, const char *name) {
	output_file << "SCALARS " << name << " float 1\n";
	output_file << "LOOKUP_TABLE default\n";
}

void output_vtk_vector_next(std::ofstream &output_file, const char *name) {
	output_file << "VECTORS " << name << " float\n";
}

template <typename T>
void output_vtk_scalar_field(std::ofstream &output_file, const ScalarField<T> &field, const char *name) {
	std::size_t nx = field.num[0], ny = field.num[1], nz = field.num[2], grid_size = nx * ny * nz;
	std::unique_ptr<uint32_t[]> vtk_scalar(new uint32_t[grid_size]);
	#pragma omp parallel for collapse(3)
	for(std::size_t k = 0; k < nz; k++) {
		for(std::size_t j = 0; j < ny; j++) {
			for(std::size_t i = 0; i < nx; i++) {
				int idx = grid_idx(i, j, k, nx, ny, nz);
				int write_idx = (k * ny * nx) + (j * nx) + i;
				vtk_scalar[write_idx] = swap_endian(static_cast<float>(field.v[idx]));
			}
		}
	}
	output_vtk_scalar_next(output_file, name);
	output_file.write(reinterpret_cast<const char*>(vtk_scalar.get()), grid_size * sizeof(uint32_t));
}

template <typename T>
void output_vtk_complex_scalar_field(std::ofstream &output_file, const ComplexScalarField<T> &field, const char *name) {
	std::size_t nx = field.num[0], ny = field.num[1], nz = field.num[2], grid_size = nx * ny * nz;
	std::unique_ptr<uint32_t[]> vtk_scalar(new uint32_t[grid_size]);
	#pragma omp parallel for collapse(3)
	for(std::size_t k = 0; k < nz; k++) {
		for(std::size_t j = 0; j < ny; j++) {
			for(std::size_t i = 0; i < nx; i++) {
				int idx = grid_idx(i, j, k, nx, ny, nz);
				int write_idx = (k * ny * nx) + (j * nx) + i;
				vtk_scalar[write_idx] = swap_endian(static_cast<float>(std::real(field.v[idx])));
			}
		}
	}
	output_vtk_scalar_next(output_file, name);
	output_file.write(reinterpret_cast<const char*>(vtk_scalar.get()), grid_size * sizeof(uint32_t));
}

template <typename T>
void output_vtk_vector_field(std::ofstream &output_file, const VectorField<T> &field, const char *name) {
	std::size_t nx = field.num[0], ny = field.num[1], nz = field.num[2], grid_size = nx * ny * nz;
	std::unique_ptr<uint32_t[]> vtk_vector(new uint32_t[3 * grid_size]);
	#pragma omp parallel for collapse(3)
	for(std::size_t k = 0; k < nz; k++) {
		for(std::size_t j = 0; j < ny; j++) {
			for(std::size_t i = 0; i < nx; i++) {
				int idx = grid_idx(i, j, k, nx, ny, nz);
				int write_idx = (k * ny * nx) + (j * nx) + i;
				vtk_vector[3 * write_idx] = swap_endian(static_cast<float>(field.x[idx]));
				vtk_vector[3 * write_idx + 1] = swap_endian(static_cast<float>(field.y[idx]));
				vtk_vector[3 * write_idx + 2] = swap_endian(static_cast<float>(field.z[idx]));
			}
		}
	}
	output_vtk_vector_next(output_file, name);
	output_file.write(reinterpret_cast<const char*>(vtk_vector.get()), 3 * grid_size * sizeof(uint32_t));
}

template void output_vtk_header<double>(std::ofstream &output_file, const ScalarField<double> &field);
template void output_vtk_header<double>(std::ofstream &output_file, const VectorField<double> &field);
template void output_vtk_scalar_field<double>(std::ofstream &output_file, const ScalarField<double> &field, const char *name);
template void output_vtk_complex_scalar_field<double>(std::ofstream &output_file, const ComplexScalarField<double> &field, const char *name);
template void output_vtk_vector_field<double>(std::ofstream &output_file, const VectorField<double> &field, const char *name);

template void output_vtk_header<float>(std::ofstream &output_file, const ScalarField<float> &field);
template void output_vtk_header<float>(std::ofstream &output_file, const VectorField<float> &field);
template void output_vtk_scalar_field<float>(std::ofstream &output_file, const ScalarField<float> &field, const char *name);
template void output_vtk_complex_scalar_field<float>(std::ofstream &output_file, const ComplexScalarField<float> &field, const char *name);
template void output_vtk_vector_field<float>(std::ofstream &output_file, const VectorField<float> &field, const char *name);