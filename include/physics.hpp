// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_H
#define PHYSICS_H

#include <span>
#include <concepts>

#include "structs.hpp"

template <std::floating_point T> void compute_lz(ScalarField<T> &lz_data, const Particles<T> &particles) noexcept;
template <std::floating_point T> void fdtd_update_b(VectorField<T> &b_data, const VectorField<T> &e_data, T dt) noexcept;
template <std::floating_point T> void fdtd_update_e(VectorField<T> &e_data, const VectorField<T> &b_data, const VectorField<T> &j_data, T dt) noexcept;
template <std::floating_point T> T total_energy(const VectorField<T> &e_data, const VectorField<T> &b_data) noexcept;
template <std::floating_point T> T divergence_e(const VectorField<T> &e_data) noexcept;
template <std::floating_point T> void initialize_ebj(VectorField<T> &e_data, VectorField<T> &b_data, VectorField<T> &j_data, std::span<const Laser<T>> lasers, T dt) noexcept;

#endif
