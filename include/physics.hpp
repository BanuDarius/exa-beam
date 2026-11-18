// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_H
#define PHYSICS_H

#include <span>
#include <concepts>

#include "structs.hpp"

template <std::floating_point T> void compute_lz(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void fdtd_update_b(VectorField<T> &b_field, VectorField<T> &e_field, T dt) noexcept;
template <std::floating_point T> void fdtd_update_e(VectorField<T> &e_field, VectorField<T> &b_field, VectorField<T> &j_field, T dt) noexcept;
template <std::floating_point T> T total_energy(VectorField<T> &e_field, VectorField<T> &b_field) noexcept;
template <std::floating_point T> T divergence_e(VectorField<T> &e_field) noexcept;
template <std::floating_point T> void initialize_ebj_fields(VectorField<T> &e_field, VectorField<T> &b_field, VectorField<T> &j_field, std::span<const Laser<T>> lasers, T dt) noexcept;

#endif
