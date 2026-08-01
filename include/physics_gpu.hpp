// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_GPU_H
#define PHYSICS_GPU_H

#include <concepts>

#include "sim_structs.hpp"
template <std::floating_point T> __global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, ParticlesView<T> particles_view);
template <std::floating_point T> __global__ void compute_u_field_gpu_kernel(ComplexScalarField<T> u_field, Laser<T> laser);
template <std::floating_point T> __global__ void compute_eb_field_gpu_kernel(VectorFieldView<T> e_field_view, VectorFieldView<T> b_field_view, ComplexScalarFieldView<T> u_field_view, Laser<T> laser, T t);
template <std::floating_point T> void compute_lz_gpu(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void compute_u_field_gpu(ComplexScalarField<T> &u_field, const Laser<T> &laser) noexcept;
template <std::floating_point T> void compute_eb_field_gpu(VectorField<T> &e_field, VectorField<T> &b_field, ComplexScalarField<T> &u_field, const Laser<T> &laser, T t) noexcept;

#endif