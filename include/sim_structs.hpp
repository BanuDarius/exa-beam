/* MIT License

Copyright (c) 2026 Banu Darius-Matei

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE. */

#ifndef SIM_STRUCTS_H
#define SIM_STRUCTS_H

#include <array>
#include <memory>
#include <complex>
#include <cassert>

#include "math_functions.hpp"

constexpr int input_file_count = 16;

template <typename T> constexpr T m_e = T(1.0);
template <typename T> constexpr T e_0 = T(-1.0);
template <typename T> constexpr T c = T(137.036);
template <typename T> constexpr T pi = T(3.14159265359);

template <typename T>
struct Parameters {
	int nx, steps, substeps;
	T tf, max_dim_mult;
	Parameters(int nx_n, int steps_n, int substeps_n, T tf_n, T max_dim_mult_n)
		: nx(nx_n), steps(steps_n), substeps(substeps_n), tf(tf_n), max_dim_mult(max_dim_mult_n) {}
	Parameters() = default;
};

template <typename T>
struct Laser {
	int p, m;
	T a0, omega, w0, k, lambda, z_r, tau, E0, psi;
	std::complex<T> zeta_x, zeta_y;
	Laser(int p_n, int m_n, T a0_n, T omega_n, T w0_multiplier, T tau_n, T psi_n, std::complex<T> zeta_x_n, std::complex<T> zeta_y_n)
		: p(p_n), m(m_n), a0(a0_n), omega(omega_n), tau(tau_n), psi(psi_n), zeta_x(zeta_x_n), zeta_y(zeta_y_n) {
		k = omega / c<T>;
		lambda = (T(2.0) * pi<T> * c<T>) / omega;
		w0 = lambda * w0_multiplier;
		z_r = pi<T> * w0 * w0 / lambda;
		E0 = omega * m_e<T> * c<T> * a0 / std::abs(e_0<T>);
	}
	Laser() = default;
};

template <typename T>
struct Particles {
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> x, y, z, ux, uy, uz, gamma;
	Particles(int nx, int ny, int nz, T r_max_n) {
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		std::size_t total = nx * ny * nz;
		x = std::make_unique_for_overwrite<T[]>(total);
		y = std::make_unique_for_overwrite<T[]>(total);
		z = std::make_unique_for_overwrite<T[]>(total);
		ux = std::make_unique_for_overwrite<T[]>(total);
		uy = std::make_unique_for_overwrite<T[]>(total);
		uz = std::make_unique_for_overwrite<T[]>(total);
		gamma = std::make_unique_for_overwrite<T[]>(total);
		#pragma omp parallel for simd collapse(3) schedule(static)
		for(int i = 0; i < nx; i++) {
			for(int j = 0; j < ny; j++) {
				for(int k = 0; k < nz; k++) {
					int idx = grid_idx(i, j, k, nx, ny, nz);
					x[idx] = interpolate(-r_max[0], r_max[0], static_cast<T>(i), static_cast<T>(nx));
					y[idx] = interpolate(-r_max[1], r_max[1], static_cast<T>(j), static_cast<T>(ny));
					z[idx] = interpolate(-r_max[2], r_max[2], static_cast<T>(k), static_cast<T>(nz));
					ux[idx] = T(0.0); uy[idx] = T(0.0); uz[idx] = T(0.0);
					gamma[idx] = T(1.0);
				}
			}
		}
	}
	Particles(const Parameters<T> &parameters, const Laser<T> &laser)
		: Particles(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult) {}
	inline std::array<T, 3> get_position(int idx) noexcept {
		std::array<T, 3> r_vec = { x[idx], y[idx], z[idx] };
		return r_vec;
	}
	inline std::array<T, 3> get_velocity(int idx) noexcept {
		std::array<T, 3> u_vec = { ux[idx], uy[idx], uz[idx] };
		return u_vec;
	}
	inline void set_position(std::array<T, 3> r_vec, int idx) noexcept {
		x[idx] = r_vec[0]; y[idx] = r_vec[1]; z[idx] = r_vec[2];
	}
	inline void set_velocity(std::array<T, 3> u_vec, int idx) noexcept {
		ux[idx] = u_vec[0]; uy[idx] = u_vec[1]; uz[idx] = u_vec[2];
	}
};

template <typename T>
struct ScalarField {
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> v;
	ScalarField(int nx, int ny, int nz, T r_max_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		v = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			v[i] = T(0.0);
	}
	ScalarField(const Parameters<T> &parameters, const Laser<T> &laser)
		: ScalarField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult) {}
	ScalarField(const ScalarField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max) {
		v = std::unique_ptr<T[]>(new T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			v[i] = other.v[i];
	}
	ScalarField &operator=(const ScalarField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] = other.v[i];
		return *this;
	}
	ScalarField &operator+=(const ScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] += other.v[i];
		return *this;
	}
	ScalarField &operator-=(const ScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] -= other.v[i];
		return *this;
	}
	ScalarField(ScalarField &&other) noexcept = default;
	ScalarField &operator=(ScalarField &&other) noexcept = default;
	~ScalarField() = default;
};

template <typename T>
struct ComplexScalarField {
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<std::complex<T>[]> v;
	ComplexScalarField(int nx, int ny, int nz, T r_max_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		v = std::make_unique_for_overwrite<std::complex<T>[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			v[i] = { T(0.0), T(0.0) };
	}
	ComplexScalarField(const Parameters<T> &parameters, const Laser<T> &laser)
		: ComplexScalarField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult) {}
	ComplexScalarField(const ComplexScalarField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max) {
		v = std::unique_ptr<std::complex<T>[]>(new std::complex<T>[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			v[i] = other.v[i];
	}
	ComplexScalarField &operator=(const ComplexScalarField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] = other.v[i];
		return *this;
	}
	ComplexScalarField &operator+=(const ComplexScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] += other.v[i];
		return *this;
	}
	ComplexScalarField &operator-=(const ComplexScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			v[i] -= other.v[i];
		return *this;
	}
	ComplexScalarField(ComplexScalarField &&other) noexcept = default;
	ComplexScalarField &operator=(ComplexScalarField &&other) noexcept = default;
	~ComplexScalarField() = default;
};

template <typename T>
struct VectorField {
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> x, y, z;
	VectorField(int nx, int ny, int nz, T r_max_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		x = std::make_unique_for_overwrite<T[]>(field_size);
		y = std::make_unique_for_overwrite<T[]>(field_size);
		z = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			x[i] = T(0.0); y[i] = T(0.0); z[i] = T(0.0);
		}
	}
	VectorField(const Parameters<T> &parameters, const Laser<T> &laser)
		: VectorField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult) {}
	VectorField(const VectorField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max) {
		x = std::unique_ptr<T[]>(new T[field_size]);
		y = std::unique_ptr<T[]>(new T[field_size]);
		z = std::unique_ptr<T[]>(new T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			x[i] = other.x[i]; y[i] = other.y[i]; z[i] = other.z[i];
		}
	}
	VectorField &operator=(const VectorField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			x[i] = other.x[i]; y[i] = other.y[i]; z[i] = other.z[i];
		}
		return *this;
	}
	VectorField &operator+=(const VectorField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			x[i] += other.x[i]; y[i] += other.y[i]; z[i] += other.z[i];
		}
		return *this;
	}
	VectorField &operator-=(const VectorField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			x[i] -= other.x[i]; y[i] -= other.y[i]; z[i] -= other.z[i];
		}
		return *this;
	}
	VectorField(VectorField &&other) noexcept = default;
	VectorField &operator=(VectorField &&other) noexcept = default;
	~VectorField() = default;
};

struct DataVTK {
	std::size_t field_size;
	std::unique_ptr<uint32_t[]> vtk_scalar, vtk_vector;
	DataVTK(int nx, int ny, int nz) {
		field_size = nx * ny * nz;
		vtk_scalar = std::make_unique_for_overwrite<uint32_t[]>(field_size);
		vtk_vector = std::make_unique_for_overwrite<uint32_t[]>(3 * field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			vtk_scalar[i] = static_cast<uint32_t>(0.0);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < 3 * field_size; i++)
			vtk_vector[i] = static_cast<uint32_t>(0.0);
	}
	DataVTK(DataVTK &&other) noexcept = default;
	DataVTK &operator=(DataVTK &&other) noexcept = default;
	~DataVTK() = default;
};

template <typename T>
struct EBVectors {
	std::array<T, 3> e, b;
	EBVectors(std::array<T, 3> e_n, std::array<T, 3> b_n) : e(e_n), b(b_n) {}
};

#endif