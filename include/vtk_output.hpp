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

#ifndef VTK_OUTPUT_H
#define VTK_OUTPUT_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <concepts>

#include "sim_structs.hpp"

inline uint32_t swap_endian(float v) noexcept {
	uint32_t data;
	std::memcpy(&data, &v, 4);
	return __builtin_bswap32(data);
}

template <std::floating_point T> void output_vtk_header(std::ofstream &output_file, ScalarField<T> &field);
template <std::floating_point T> void output_vtk_header(std::ofstream &output_file, VectorField<T> &field);
void output_vtk_scalar_next(std::ofstream &output_file, const std::string &name);
void output_vtk_vector_next(std::ofstream &output_file, const std::string &name);
template <std::floating_point T> void output_vtk_scalar_field(std::ofstream &output_file, DataVTK &data_vtk, ScalarField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_complex_scalar_field(std::ofstream &output_file, DataVTK &data_vtk, ComplexScalarField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_vector_field(std::ofstream &output_file, DataVTK &data_vtk, VectorField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_particles(std::ofstream &output_file, DataVTK &data_vtk, Particles<T> &particles);

#endif