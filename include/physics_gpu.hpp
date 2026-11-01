// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_GPU_H
#define PHYSICS_GPU_H

#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> __global__ void compute_lz_gpu_kernel(ScalarFieldView<T> lz_view, ParticlesView<T> particles_view);
template <std::floating_point T> void compute_lz_gpu(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;

#endif