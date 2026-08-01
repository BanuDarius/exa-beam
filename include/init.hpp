// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef INIT_H
#define INIT_H

#include <cstring>
#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> void read_input_file(const std::string &parameters_filename, const std::string &laser_filename, Parameters<T> &parameters, Laser<T> &laser);

#endif