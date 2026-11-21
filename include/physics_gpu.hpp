// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_GPU_H
#define PHYSICS_GPU_H

#include <concepts>

#include "structs.hpp"

template <std::floating_point T> __global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, const ParticlesView<T> particles_view);
template <std::floating_point T> __global__ void fdtd_update_b_gpu_kernel(VectorFieldView<T> b, const VectorFieldView<T> e, T dt);
template <std::floating_point T> __global__ void fdtd_update_e_gpu_kernel(VectorFieldView<T> e, const VectorFieldView<T> b, const VectorFieldView<T> J, T dt);
template <std::floating_point T> __global__ void total_energy_gpu_kernel(const VectorFieldView<T> e, const VectorFieldView<T> b, T *d_energy);
template <std::floating_point T> __global__ void initialize_ebj_gpu_kernel(VectorFieldView<T> e, VectorFieldView<T> b, VectorFieldView<T> J, __grid_constant__ const DeviceLasers<T> lasers, T dt);
template <std::floating_point T> void compute_lz_gpu(ScalarField<T> &lz_field, const Particles<T> &particles) noexcept;
template <std::floating_point T> void fdtd_update_b_gpu(VectorField<T> &b_data, const VectorField<T> &e_data, T dt) noexcept;
template <std::floating_point T> void fdtd_update_e_gpu(VectorField<T> &e_data, const VectorField<T> &b_data, const VectorField<T> &j_data, T dt) noexcept;
template <std::floating_point T> T total_energy_gpu(const VectorField<T> &e_data, const VectorField<T> &b_data) noexcept;
template <std::floating_point T> T divergence_e_gpu(const VectorField<T> &e_data) noexcept;
template <std::floating_point T> void initialize_ebj_gpu(VectorField<T> &e_data, VectorField<T> &b_data, VectorField<T> &j_data, const DeviceLasers<T> &lasers, T dt) noexcept;

#endif
