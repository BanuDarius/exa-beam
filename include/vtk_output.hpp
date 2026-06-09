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

#include <cstdio>
#include <cstdint>
#include <cstring>

#include "sim_structs.hpp"

inline uint32_t swap_endian(float v) {
	uint32_t data;
	std::memcpy(&data, &v, 4);
	return __builtin_bswap32(data);
}

template <typename T> void output_vtk_header(std::FILE *out, const ScalarField<T> &field);
template <typename T> void output_vtk_header(std::FILE *out, const VectorField<T> &field);
void output_vtk_scalar_next(std::FILE *out, const char *name);
void output_vtk_vector_next(std::FILE *out, const char *name);
template <typename T> void output_vtk_scalar_field(std::FILE *out, const ScalarField<T> &field, const char *name);
template <typename T> void output_vtk_complex_scalar_field(std::FILE *out, const ComplexScalarField<T> &field, const char *name);
template <typename T> void output_vtk_vector_field(std::FILE *out, const VectorField<T> &field, const char *name);

#endif