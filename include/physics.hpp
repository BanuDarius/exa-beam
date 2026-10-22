// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_H
#define PHYSICS_H

#include <span>
#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> void compute_lz(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void compute_u_field(ComplexScalarField<T> &u_field, std::span<const Laser<T>> lasers) noexcept;
template <std::floating_point T> void compute_e_field(VectorField<T> &e_field, std::span<const Laser<T>> lasers, T t) noexcept;
template <std::floating_point T> void compute_b_field(VectorField<T> &b_field, std::span<const Laser<T>> lasers, T t) noexcept;

#endif