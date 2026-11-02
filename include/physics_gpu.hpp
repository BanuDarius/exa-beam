// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_GPU_H
#define PHYSICS_GPU_H

#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> __global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, ParticlesView<T> particles_view);
template <std::floating_point T> __global__ void fdtd_update_b_gpu_kernel(VectorFieldView<T> b, VectorFieldView<T> e, T dt);
template <std::floating_point T> __global__ void fdtd_update_e_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, T dt);
template <std::floating_point T> __global__ void compute_total_energy_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, T *d_energy);
template <std::floating_point T> __global__ void initialize_eb_fields_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, __grid_constant__ const DeviceLasers<T> lasers, T dt);
template <std::floating_point T> void compute_lz_gpu(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void fdtd_update_b_gpu(VectorField<T> &b_field, VectorField<T> &e_field, T dt) noexcept;
template <std::floating_point T> void fdtd_update_e_gpu(VectorField<T> &e_field, VectorField<T> &b_field, T dt) noexcept;
template <std::floating_point T> T compute_total_energy_gpu(VectorField<T> &e_field, VectorField<T> &b_field) noexcept;
template <std::floating_point T> void initialize_eb_fields_gpu(VectorField<T> &e_field, VectorField<T> &b_field, const DeviceLasers<T> &lasers, T dt) noexcept;

#endif