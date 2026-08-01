// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

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

template <std::floating_point T> void output_vtk_header(std::ofstream &output_file, const ScalarField<T> &field);
template <std::floating_point T> void output_vtk_header(std::ofstream &output_file, const VectorField<T> &field);
void output_vtk_scalar_next(std::ofstream &output_file, const std::string &name);
void output_vtk_vector_next(std::ofstream &output_file, const std::string &name);
template <std::floating_point T> void output_vtk_scalar_field(std::ofstream &output_file, const DataVTK &data_vtk, const ScalarField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_complex_scalar_field(std::ofstream &output_file, const DataVTK &data_vtk, const ComplexScalarField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_vector_field(std::ofstream &output_file,const DataVTK &data_vtk, const VectorField<T> &field, const std::string &name);
template <std::floating_point T> void output_vtk_particles(std::ofstream &output_file, const DataVTK &data_vtk, const Particles<T> &particles);

#endif