// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef INIT_H
#define INIT_H

#include <vector>
#include <cstring>
#include <concepts>

#include "sim_structs.hpp"

template <std::floating_point T> void read_input_file(const std::string &parameters_filename, Parameters<T> &parameters);
template <std::floating_point T> void read_lasers_file(const std::string &lasers_filename, std::vector<Laser<T>> &lasers, const Parameters<T> &parameters);

#endif