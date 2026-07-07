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
#include <cstdint>
#include <cassert>
#include <concepts>

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include "structs_view.hpp"
#include "math_functions.hpp"

constexpr int input_file_count = 17;

template <std::floating_point T> constexpr T m_e = T(1.0);
template <std::floating_point T> constexpr T e_0 = T(-1.0);
template <std::floating_point T> constexpr T c = T(137.036);
template <std::floating_point T> constexpr T pi = T(3.14159265359);

template <std::floating_point T>
struct Parameters {
	int nx, steps, substeps;
	T tf, max_dim_mult;
	bool use_gpu;
	Parameters(int nx_n, int steps_n, int substeps_n, T tf_n, T max_dim_mult_n, bool use_gpu_n)
		: nx(nx_n), steps(steps_n), substeps(substeps_n), tf(tf_n), max_dim_mult(max_dim_mult_n), use_gpu(use_gpu_n) {}
	Parameters() = default;
};

template <std::floating_point T>
struct Laser {
	int p, m;
	T a0, omega, w0, k, lambda, z_r, tau, E0, psi;
	cuda::std::complex<T> zeta_x, zeta_y;
	Laser(int p_n, int m_n, T a0_n, T omega_n, T w0_multiplier, T tau_n, T psi_n, cuda::std::complex<T> zeta_x_n, cuda::std::complex<T> zeta_y_n)
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
struct CUDAMemoryAdmin {
	void operator()(T* ptr) noexcept {
		if(ptr) cudaFree(ptr);
	}
};

template <std::floating_point T>
struct Particles {
	bool use_gpu;
	std::size_t particle_num;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> h_x, h_y, h_z, h_ux, h_uy, h_uz, h_gamma;
	std::unique_ptr<T[], CUDAMemoryAdmin<T>> d_x, d_y, d_z, d_ux, d_uy, d_uz, d_gamma;
	Particles(int nx, int ny, int nz, T r_max_n, bool use_gpu_n) : use_gpu(use_gpu_n) {
		particle_num = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		h_x = std::make_unique_for_overwrite<T[]>(particle_num);
		h_y = std::make_unique_for_overwrite<T[]>(particle_num);
		h_z = std::make_unique_for_overwrite<T[]>(particle_num);
		h_ux = std::make_unique_for_overwrite<T[]>(particle_num);
		h_uy = std::make_unique_for_overwrite<T[]>(particle_num);
		h_uz = std::make_unique_for_overwrite<T[]>(particle_num);
		h_gamma = std::make_unique_for_overwrite<T[]>(particle_num);
		#pragma omp parallel for simd collapse(3) schedule(static)
		for(int i = 0; i < nx; i++) {
			for(int j = 0; j < ny; j++) {
				for(int k = 0; k < nz; k++) {
					int idx = grid_idx(i, j, k, nx, ny, nz);
					h_x[idx] = interpolate(-r_max[0], r_max[0], static_cast<T>(i), static_cast<T>(nx));
					h_y[idx] = interpolate(-r_max[1], r_max[1], static_cast<T>(j), static_cast<T>(ny));
					h_z[idx] = interpolate(-r_max[2], r_max[2], static_cast<T>(k), static_cast<T>(nz));
					h_ux[idx] = T(0.0); h_uy[idx] = T(0.0); h_uz[idx] = T(0.0);
					h_gamma[idx] = T(1.0);
				}
			}
		}
		if(use_gpu) {
			T *raw_x = nullptr, *raw_y = nullptr, *raw_z = nullptr;
			T *raw_ux = nullptr, *raw_uy = nullptr, *raw_uz = nullptr, *raw_gamma = nullptr;
			
			cudaMalloc(&raw_x, particle_num * sizeof(T));
			cudaMalloc(&raw_y, particle_num * sizeof(T));
			cudaMalloc(&raw_z, particle_num * sizeof(T));
			cudaMalloc(&raw_ux, particle_num * sizeof(T));
			cudaMalloc(&raw_uy, particle_num * sizeof(T));
			cudaMalloc(&raw_uz, particle_num * sizeof(T));
			cudaMalloc(&raw_gamma, particle_num * sizeof(T));
			
			d_x.reset(raw_x); d_y.reset(raw_y); d_z.reset(raw_z);
			d_ux.reset(raw_ux); d_uy.reset(raw_uy); d_uz.reset(raw_uz);
			d_gamma.reset(raw_gamma);
			
			transfer_data_cpu_to_gpu();
		}
	}
	inline void transfer_data_cpu_to_gpu() noexcept {
		cudaMemcpy(d_x.get(), h_x.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_y.get(), h_y.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_z.get(), h_z.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_ux.get(), h_ux.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uy.get(), h_uy.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_uz.get(), h_uz.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
		cudaMemcpy(d_gamma.get(), h_gamma.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice);
	}
	inline void transfer_data_gpu_to_cpu() noexcept {
		cudaMemcpy(h_x.get(), d_x.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_y.get(), d_y.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_z.get(), d_z.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_ux.get(), d_ux.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_uy.get(), d_uy.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_uz.get(), d_uz.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_gamma.get(), d_gamma.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost);
	}
	ParticlesView<T> get_cpu_view() const noexcept {
		return ParticlesView<T>(h_x.get(), h_y.get(), h_z.get(), h_ux.get(), h_uy.get(), h_uz.get(), h_gamma.get());
	}
	ParticlesView<T> get_gpu_view() const noexcept {
		return ParticlesView<T>(d_x.get(), d_y.get(), d_z.get(), d_ux.get(), d_uy.get(), d_uz.get(), d_gamma.get());
	}
	Particles(const Parameters<T> &parameters, const Laser<T> &laser)
		: Particles(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult, parameters.use_gpu) {}
	Particles(Particles &&other) noexcept = default;
	Particles &operator=(Particles &&other) noexcept = default;
	~Particles() = default;
};

template <std::floating_point T>
struct ScalarField {
	bool use_gpu;
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> h_v;
	std::unique_ptr<T[], CUDAMemoryAdmin<T>> d_v;
	ScalarField(int nx, int ny, int nz, T r_max_n, bool use_gpu_n) : use_gpu(use_gpu_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		h_v = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = T(0.0);
		if(use_gpu) {
			T *raw_v = nullptr;
			
			cudaMalloc(&raw_v, field_size * sizeof(T));
			
			d_v.reset(raw_v);
			transfer_data_cpu_to_gpu();
		}
	}
	ScalarField(const ScalarField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max), use_gpu(other.use_gpu) {
		h_v = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = other.h_v[i];
		if(use_gpu) {
			T *raw_v = nullptr;
			
			cudaMalloc(&raw_v, field_size * sizeof(T));
			
			d_v.reset(raw_v);
			cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
		}
	}
	ScalarField &operator=(const ScalarField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] = other.h_v[i];
		if(use_gpu)
			cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
		return *this;
	}
	ScalarField &operator+=(const ScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] += other.h_v[i];
		return *this;
	}
	ScalarField &operator-=(const ScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] -= other.h_v[i];
		return *this;
	}
	inline void transfer_data_cpu_to_gpu() noexcept {
		cudaMemcpy(d_v.get(), h_v.get(), field_size * sizeof(T), cudaMemcpyHostToDevice);
	}
	inline void transfer_data_gpu_to_cpu() noexcept {
		cudaMemcpy(h_v.get(), d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost);
	}
	ScalarFieldView<T> get_cpu_view() const noexcept {
		return ScalarFieldView<T>(h_v.get());
	}
	ScalarFieldView<T> get_gpu_view() const noexcept {
		return ScalarFieldView<T>(d_v.get());
	}
	ScalarField(const Parameters<T> &parameters, const Laser<T> &laser)
		: ScalarField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult, parameters.use_gpu) {}
	ScalarField(ScalarField &&other) noexcept = default;
	ScalarField &operator=(ScalarField &&other) noexcept = default;
	~ScalarField() = default;
};

template <std::floating_point T>
struct ComplexScalarField {
	bool use_gpu;
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<cuda::std::complex<T>[]> h_v;
	std::unique_ptr<cuda::std::complex<T>[], CUDAMemoryAdmin<cuda::std::complex<T>>> d_v;
	ComplexScalarField(int nx, int ny, int nz, T r_max_n, bool use_gpu_n) : use_gpu(use_gpu_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		h_v = std::make_unique_for_overwrite<cuda::std::complex<T>[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = { T(0.0), T(0.0) };
		if(use_gpu) {
			cuda::std::complex<T> *raw_v = nullptr;
			
			cudaMalloc(&raw_v, field_size * sizeof(cuda::std::complex<T>));
			
			d_v.reset(raw_v);
			transfer_data_cpu_to_gpu();
		}
	}
	ComplexScalarField(const ComplexScalarField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max) {
		h_v = std::make_unique_for_overwrite<cuda::std::complex<T>[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = other.h_v[i];
		if(use_gpu) {
			T *raw_v = nullptr;
			
			cudaMalloc(&raw_v, field_size * sizeof(cuda::std::complex<T>));
			
			d_v.reset(raw_v);
			cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(cuda::std::complex<T>), cudaMemcpyDeviceToDevice);
		}
	}
	ComplexScalarField &operator=(const ComplexScalarField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] = other.h_v[i];
		if(use_gpu)
			cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(cuda::std::complex<T>), cudaMemcpyDeviceToDevice);
		return *this;
	}
	ComplexScalarField &operator+=(const ComplexScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] += other.h_v[i];
		return *this;
	}
	ComplexScalarField &operator-=(const ComplexScalarField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++)
			h_v[i] -= other.h_v[i];
		return *this;
	}
	inline void transfer_data_cpu_to_gpu() noexcept {
		cudaMemcpy(d_v.get(), h_v.get(), field_size * sizeof(cuda::std::complex<T>), cudaMemcpyHostToDevice);
	}
	inline void transfer_data_gpu_to_cpu() noexcept {
		cudaMemcpy(h_v.get(), d_v.get(), field_size * sizeof(cuda::std::complex<T>), cudaMemcpyDeviceToHost);
	}
	ComplexScalarFieldView<T> get_cpu_view() const noexcept {
		return ComplexScalarFieldView<T>(h_v.get());
	}
	ComplexScalarFieldView<T> get_gpu_view() const noexcept {
		return ComplexScalarFieldView<T>(d_v.get());
	}
	ComplexScalarField(const Parameters<T> &parameters, const Laser<T> &laser)
		: ComplexScalarField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult, parameters.use_gpu) {}
	ComplexScalarField(ComplexScalarField &&other) noexcept = default;
	ComplexScalarField &operator=(ComplexScalarField &&other) noexcept = default;
	~ComplexScalarField() = default;
};

template <std::floating_point T>
struct VectorField {
	bool use_gpu;
	std::size_t field_size;
	std::array<int, 3> num;
	std::array<T, 3> r_max;
	std::unique_ptr<T[]> h_x, h_y, h_z;
	std::unique_ptr<T[], CUDAMemoryAdmin<T>> d_x, d_y, d_z;
	VectorField(int nx, int ny, int nz, T r_max_n, bool use_gpu_n) : use_gpu(use_gpu_n) {
		field_size = nx * ny * nz;
		num = { nx, ny, nz };
		r_max = { r_max_n, r_max_n, r_max_n };
		h_x = std::make_unique_for_overwrite<T[]>(field_size);
		h_y = std::make_unique_for_overwrite<T[]>(field_size);
		h_z = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			h_x[i] = T(0.0); h_y[i] = T(0.0); h_z[i] = T(0.0);
		}
		if(use_gpu) {
			T *raw_x = nullptr, *raw_y = nullptr, *raw_z = nullptr;
			
			cudaMalloc(&raw_x, field_size * sizeof(T));
			cudaMalloc(&raw_y, field_size * sizeof(T));
			cudaMalloc(&raw_z, field_size * sizeof(T));
			
			d_x.reset(raw_x); d_y.reset(raw_y); d_z.reset(raw_z);
			transfer_data_cpu_to_gpu();
		}
	}
	VectorField(const VectorField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max) {
		h_x = std::make_unique_for_overwrite<T[]>(field_size);
		h_y = std::make_unique_for_overwrite<T[]>(field_size);
		h_z = std::make_unique_for_overwrite<T[]>(field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			h_x[i] = other.h_x[i]; h_y[i] = other.h_y[i]; h_z[i] = other.h_z[i];
		}
		if(use_gpu) {
			T *raw_x = nullptr, *raw_y = nullptr, *raw_z = nullptr;
			
			cudaMalloc(&raw_x, field_size * sizeof(T));
			cudaMalloc(&raw_y, field_size * sizeof(T));
			cudaMalloc(&raw_z, field_size * sizeof(T));
			
			d_x.reset(raw_x); d_y.reset(raw_y); d_z.reset(raw_z);
			cudaMemcpy(d_x.get(), other.d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_y.get(), other.d_y.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_z.get(), other.d_z.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
		}
	}
	VectorField &operator=(const VectorField &other) {
		if(this == &other) return *this;
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			h_x[i] = other.h_x[i]; h_y[i] = other.h_y[i]; h_z[i] = other.h_z[i];
		}
		if(use_gpu) {
			cudaMemcpy(d_x.get(), other.d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_y.get(), other.d_y.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
			cudaMemcpy(d_z.get(), other.d_z.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice);
		}
		return *this;
	}
	VectorField &operator+=(const VectorField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			h_x[i] += other.h_x[i]; h_y[i] += other.h_y[i]; h_z[i] += other.h_z[i];
		}
		return *this;
	}
	VectorField &operator-=(const VectorField &other) {
		assert(field_size == other.field_size && "FIELD SIZES DO NOT MATCH!");
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < other.field_size; i++) {
			h_x[i] -= other.h_x[i]; h_y[i] -= other.h_y[i]; h_z[i] -= other.h_z[i];
		}
		return *this;
	}
	inline void transfer_data_cpu_to_gpu() noexcept {
		cudaMemcpy(d_x.get(), h_x.get(), field_size * sizeof(T), cudaMemcpyHostToDevice);
	}
	inline void transfer_data_gpu_to_cpu() noexcept {
		cudaMemcpy(h_x.get(), d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost);
	}
	VectorFieldView<T> get_cpu_view() const noexcept {
		return VectorFieldView<T>(h_x.get(), h_y.get(), h_z.get());
	}
	VectorFieldView<T> get_gpu_view() const noexcept {
		return VectorFieldView<T>(d_x.get(), d_y.get(), d_z.get());
	}
	VectorField(const Parameters<T> &parameters, const Laser<T> &laser)
		: VectorField(parameters.nx, parameters.nx, parameters.nx, laser.w0 * parameters.max_dim_mult, parameters.use_gpu) {}
	VectorField(VectorField &&other) noexcept = default;
	VectorField &operator=(VectorField &&other) noexcept = default;
	~VectorField() = default;
};

struct DataVTK {
	bool use_gpu;
	std::size_t field_size;
	std::unique_ptr<uint32_t[]> vtk_scalar, vtk_vector;
	DataVTK(int nx, int ny, int nz, bool use_gpu_n) : use_gpu(use_gpu_n) {
		field_size = nx * ny * nz;
		vtk_scalar = std::make_unique_for_overwrite<uint32_t[]>(field_size);
		vtk_vector = std::make_unique_for_overwrite<uint32_t[]>(3 * field_size);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			vtk_scalar[i] = 0u;
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < 3 * field_size; i++)
			vtk_vector[i] = 0u;
	}
	DataVTK(const Parameters<double> &parameters)
		: DataVTK(parameters.nx, parameters.nx, parameters.nx, parameters.use_gpu) {}
	DataVTK(const Parameters<float> &parameters)
		: DataVTK(parameters.nx, parameters.nx, parameters.nx, parameters.use_gpu) {}
	DataVTK(DataVTK &&other) noexcept = default;
	DataVTK &operator=(DataVTK &&other) noexcept = default;
	~DataVTK() = default;
};

template <std::floating_point T>
struct EBVectors {
	std::array<T, 3> e, b;
	EBVectors(std::array<T, 3> e_n, std::array<T, 3> b_n) : e(e_n), b(b_n) {}
};

#endif