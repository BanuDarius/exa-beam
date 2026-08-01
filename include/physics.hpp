// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef PHYSICS_H
#define PHYSICS_H

#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> void compute_lz(ScalarField<T> &lz_field, Particles<T> &particles) noexcept;
template <std::floating_point T> void compute_u_field(ComplexScalarField<T> &u_field, const Laser<T> &laser) noexcept;
template <std::floating_point T> void compute_eb_field(VectorField<T> &e_field, VectorField<T> &b_field, const ComplexScalarField<T> &u_field, const Laser<T> &laser, T t) noexcept;

#endif