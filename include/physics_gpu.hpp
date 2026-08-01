// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_GPU_H
#define PHYSICS_GPU_H

#include <span>
#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> __global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, ParticlesView<T> particles_view);
template <std::floating_point T> __global__ void compute_u_field_gpu_kernel(ComplexScalarField<T> u_field);
template <std::floating_point T> __global__ void compute_eb_field_gpu_kernel(VectorFieldView<T> e_field_view, VectorFieldView<T> b_field_view, T t);
template <std::floating_point T> void compute_lz_gpu(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void compute_u_field_gpu(ComplexScalarField<T> &u_field) noexcept;
template <std::floating_point T> void compute_eb_field_gpu(VectorField<T> &e_field, VectorField<T> &b_field, T t) noexcept;

#endif