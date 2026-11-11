// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef SIM_STRUCTS_H
#define SIM_STRUCTS_H

#include <span>
#include <numbers>

#include <cuda_runtime.h>
#include <cuda/std/array>
#include <cuda/std/complex>

#include "cuda_extra.hpp"
#include "structs_view.hpp"
#include "math_functions.hpp"

constexpr int max_lasers = 16;
constexpr int input_file_count = 12;
constexpr int input_laser_count = 16;

template <std::floating_point T> constexpr T m_e = T(1.0);
template <std::floating_point T> constexpr T e_0 = T(-1.0);
template <std::floating_point T> constexpr T c = T(137.036);
template <std::floating_point T> constexpr T c2 = c<T> * c<T>;
template <std::floating_point T> constexpr T pi = std::numbers::pi_v<T>;

struct DataVTK;
template <std::floating_point T> struct Laser;
template <std::floating_point T> struct EBVectors;
template <std::floating_point T> struct Parameters;
template <std::floating_point T> struct ScalarField;
template <std::floating_point T> struct VectorField;
template <std::floating_point T> struct ComplexScalarField;

template <std::floating_point T>
struct Parameters {
	T tf;
	cuda::std::array<int, 3> num;
	int steps, substeps, laser_count;
	Vector3<T> max_dim_mult;
	bool use_gpu, output_laser_fields;
	Parameters(T tf_new, cuda::std::array<int, 3> num_new, int steps_new, int substeps_new, int laser_count_new, Vector3<T> max_dim_mult_new, bool use_gpu_new, bool output_laser_fields_new)
		: tf(tf_new), num(num_new), steps(steps_new), substeps(substeps_new), laser_count(laser_count_new), max_dim_mult(max_dim_mult_new), use_gpu(use_gpu_new), output_laser_fields(output_laser_fields_new) {}
	Parameters() noexcept = default;
};

template <std::floating_point T>
struct Laser {
	int p, m;
	T a0, omega, w0, k, lambda, z_r, tau, E0, psi;
	cuda::std::complex<T> zeta_x, zeta_y;
	Vector3<T> ex_prime, ey_prime, ez_prime, r0;
	Laser(int p_new, int m_new, T a0_new, T omega_new, T w0_mult, T tau_new, T psi_new, cuda::std::complex<T> zeta_x_new, cuda::std::complex<T> zeta_y_new, Vector3<T> r0_new, T phi, T theta)
		: p(p_new), m(m_new), a0(a0_new), omega(omega_new), tau(tau_new), psi(psi_new), zeta_x(zeta_x_new), zeta_y(zeta_y_new) {
		k = omega / c<T>;
		lambda = (T(2.0) * pi<T> * c<T>) / omega;
		w0 = lambda * w0_mult;
		z_r = pi<T> * w0 * w0 / lambda;
		E0 = omega * m_e<T> * c<T> * a0 / std::abs(e_0<T>);
		r0 = r0_new * w0;
		ex_prime = {
			cuda::std::cos(theta) * cuda::std::cos(phi),
			cuda::std::cos(theta) * cuda::std::sin(phi),
			- cuda::std::sin(theta)
		};
		ey_prime = {
			- cuda::std::sin(phi),
			cuda::std::cos(phi),
			T(0.0)
		};
		ez_prime = {
			cuda::std::sin(theta) * cuda::std::cos(phi),
			cuda::std::sin(theta) * cuda::std::sin(phi),
			cuda::std::cos(theta)
		};
	}
	__device__ __host__ Vector3<T> pos_global_to_local(Vector3<T> r_vec_global) const noexcept {
		r_vec_global -= r0;
		Vector3<T> r_vec_local = {
			dot(ex_prime, r_vec_global),
			dot(ey_prime, r_vec_global),
			dot(ez_prime, r_vec_global)
		};
		return r_vec_local;
	}
	__device__ __host__ Vector3<T> vec_local_to_global(Vector3<T> vec_local) const noexcept {
		Vector3<T> ex_prime_vec = ex_prime * vec_local[0];
		Vector3<T> ey_prime_vec = ey_prime * vec_local[1];
		Vector3<T> ez_prime_vec = ez_prime * vec_local[2];
		Vector3<T> vec_global = ex_prime_vec + ey_prime_vec + ez_prime_vec;
		return vec_global;
	}
	Laser() noexcept = default;
};

template <std::floating_point T>
struct DeviceLasers {
	int laser_count;
	cuda::std::array<Laser<T>, max_lasers> d_lasers;
	__device__ const Laser<T> &operator[](int idx) const noexcept {
		return d_lasers[idx];
	}
	DeviceLasers(std::span<const Laser<T>> h_lasers) : laser_count(h_lasers.size()) {
		for(int i = 0; i < laser_count; i++)
			d_lasers[i] = h_lasers[i];
	}
	DeviceLasers() noexcept = default;
};

template <std::floating_point T>
struct Particles {
	cuda::std::array<int, 3> num;
	Vector3<T> r_max;
	bool use_gpu;
	std::size_t particle_num;
	std::unique_ptr<T[], HostMemory<T>> h_x, h_y, h_z, h_ux, h_uy, h_uz, h_gamma;
	std::unique_ptr<T[], DeviceMemory<T>> d_x, d_y, d_z, d_ux, d_uy, d_uz, d_gamma;
	Particles(cuda::std::array<int, 3> num_new, Vector3<T> r_max_new, bool use_gpu_new) : num(num_new), r_max(r_max_new), use_gpu(use_gpu_new) {
		particle_num = num[0] * num[1] * num[2];
		h_x.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_y.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_z.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_ux.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_uy.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_uz.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		h_gamma.reset(new (static_cast<std::align_val_t>(mem_align)) T[particle_num]);
		#pragma omp parallel for simd collapse(3) schedule(static)
		for(int k = 0; k < num[2]; k++)
			for(int j = 0; j < num[1]; j++)
				for(int i = 0; i < num[0]; i++) {
					std::size_t idx = grid_idx(i, j, k, num[0], num[1], num[2]);
					h_x[idx] = interpolate(-r_max[0], r_max[0], static_cast<T>(i), static_cast<T>(num[0]));
					h_y[idx] = interpolate(-r_max[1], r_max[1], static_cast<T>(j), static_cast<T>(num[1]));
					h_z[idx] = interpolate(-r_max[2], r_max[2], static_cast<T>(k), static_cast<T>(num[2]));
					h_ux[idx] = T(0.0); h_uy[idx] = T(0.0); h_uz[idx] = T(0.0);
					h_gamma[idx] = T(1.0);
				}
		if(use_gpu) {
			T *raw_d_x = nullptr, *raw_d_y = nullptr, *raw_d_z = nullptr;
			T *raw_d_ux = nullptr, *raw_d_uy = nullptr, *raw_d_uz = nullptr, *raw_d_gamma = nullptr;
			
			CUDA_CHECK(cudaMalloc(&raw_d_x, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_y, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_z, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_ux, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_uy, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_uz, particle_num * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_gamma, particle_num * sizeof(T)));
			
			d_x.reset(raw_d_x); d_y.reset(raw_d_y); d_z.reset(raw_d_z);
			d_ux.reset(raw_d_ux); d_uy.reset(raw_d_uy); d_uz.reset(raw_d_uz);
			d_gamma.reset(raw_d_gamma);
			
			CUDA_CHECK(cudaHostRegister(h_x.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_y.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_z.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_ux.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_uy.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_uz.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_gamma.get(), particle_num * sizeof(T), cudaHostRegisterDefault));
			
			transfer_data_cpu_to_gpu(cudaStreamDefault);
		}
	}
	void transfer_data_cpu_to_gpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(d_x.get(), h_x.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_y.get(), h_y.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_z.get(), h_z.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_ux.get(), h_ux.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_uy.get(), h_uy.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_uz.get(), h_uz.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_gamma.get(), h_gamma.get(), particle_num * sizeof(T), cudaMemcpyHostToDevice, stream));
	}
	void transfer_data_gpu_to_cpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(h_x.get(), d_x.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_y.get(), d_y.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_z.get(), d_z.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_ux.get(), d_ux.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_uy.get(), d_uy.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_uz.get(), d_uz.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_gamma.get(), d_gamma.get(), particle_num * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}
	ParticlesView<T> get_cpu_view() const noexcept {
		return ParticlesView<T>(h_x.get(), h_y.get(), h_z.get(), h_ux.get(), h_uy.get(), h_uz.get(), h_gamma.get(), num, r_max, particle_num);
	}
	ParticlesView<T> get_gpu_view() const noexcept {
		return ParticlesView<T>(d_x.get(), d_y.get(), d_z.get(), d_ux.get(), d_uy.get(), d_uz.get(), d_gamma.get(), num, r_max, particle_num);
	}
	Particles(const Parameters<T> &parameters, const Laser<T> &laser)
		: Particles(parameters.num, parameters.max_dim_mult * laser.w0, parameters.use_gpu) {}
	Particles(Particles &&other) noexcept = default;
	Particles &operator=(Particles &&other) noexcept = default;
	~Particles() = default;
};

template <std::floating_point T>
struct ScalarField {
	cuda::std::array<int, 3> num;
	Vector3<T> r_max;
	bool use_gpu;
	std::size_t field_size;
	std::unique_ptr<T[], HostMemory<T>> h_v;
	std::unique_ptr<T[], DeviceMemory<T>> d_v;
	ScalarField(cuda::std::array<int, 3> num_new, Vector3<T> r_max_new, bool use_gpu_new) : num(num_new), r_max(r_max_new), use_gpu(use_gpu_new) {
		field_size = num[0] * num[1] * num[2];
		h_v.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = T(0.0);
		if(use_gpu) {
			T *raw_d_v = nullptr;
			CUDA_CHECK(cudaMalloc(&raw_d_v, field_size * sizeof(T)));
			d_v.reset(raw_d_v);
			
			CUDA_CHECK(cudaHostRegister(h_v.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			transfer_data_cpu_to_gpu(cudaStreamDefault);
		}
	}
	ScalarField(const ScalarField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max), use_gpu(other.use_gpu) {
		h_v.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			h_v[i] = other.h_v[i];
		if(use_gpu) {
			T *raw_d_v = nullptr;
			CUDA_CHECK(cudaMalloc(&raw_d_v, field_size * sizeof(T)));
			d_v.reset(raw_d_v);
			
			CUDA_CHECK(cudaHostRegister(h_v.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
	}
	ScalarField &operator=(const ScalarField &other) {
		if(this == &other) return *this;
		if(!use_gpu) {
			#pragma omp parallel for simd schedule(static)
			for(std::size_t i = 0; i < other.field_size; i++)
				h_v[i] = other.h_v[i];
		} else
			CUDA_CHECK(cudaMemcpy(d_v.get(), other.d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
		return *this;
	}
	void transfer_data_cpu_to_gpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(d_v.get(), h_v.get(), field_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	}
	void transfer_data_gpu_to_cpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(h_v.get(), d_v.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}
	ScalarFieldView<T> get_cpu_view() const noexcept {
		return ScalarFieldView<T>(h_v.get(), num, r_max, field_size);
	}
	ScalarFieldView<T> get_gpu_view() const noexcept {
		return ScalarFieldView<T>(d_v.get(), num, r_max, field_size);
	}
	ScalarField(const Parameters<T> &parameters, const Laser<T> &laser)
		: ScalarField(parameters.num, parameters.max_dim_mult * laser.w0, parameters.use_gpu) {}
	ScalarField(ScalarField &&other) noexcept = default;
	ScalarField &operator=(ScalarField &&other) noexcept = default;
	~ScalarField() = default;
};

template <std::floating_point T>
struct VectorField {
	cuda::std::array<int, 3> num;
	Vector3<T> r_max, inv_dx, dx;
	bool use_gpu;
	std::size_t field_size;
	std::unique_ptr<T[], HostMemory<T>> h_x, h_y, h_z;
	std::unique_ptr<T[], DeviceMemory<T>> d_x, d_y, d_z;
	VectorField(cuda::std::array<int, 3> num_new, Vector3<T> r_max_new, bool use_gpu_new) : num(num_new), r_max(r_max_new), use_gpu(use_gpu_new) {
		field_size = num[0] * num[1] * num[2];
		dx = {
			T(2.0) * r_max[0] / static_cast<T>(num[0] - 1),
			T(2.0) * r_max[1] / static_cast<T>(num[1] - 1),
			T(2.0) * r_max[2] / static_cast<T>(num[2] - 1)
		};
		inv_dx = { T(1.0) / dx[0], T(1.0) / dx[1], T(1.0) / dx[2] };
		h_x.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		h_y.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		h_z.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			h_x[i] = T(0.0); h_y[i] = T(0.0); h_z[i] = T(0.0);
		}
		if(use_gpu) {
			T *raw_d_x = nullptr, *raw_d_y = nullptr, *raw_d_z = nullptr;
			CUDA_CHECK(cudaMalloc(&raw_d_x, field_size * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_y, field_size * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_d_z, field_size * sizeof(T)));
			d_x.reset(raw_d_x); d_y.reset(raw_d_y); d_z.reset(raw_d_z);
			
			CUDA_CHECK(cudaHostRegister(h_x.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_y.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_z.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			transfer_data_cpu_to_gpu(cudaStreamDefault);
		}
	}
	VectorField(const VectorField &other)
		: field_size(other.field_size), num(other.num), r_max(other.r_max), use_gpu(other.use_gpu) {
		h_x.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		h_y.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		h_z.reset(new (static_cast<std::align_val_t>(mem_align)) T[field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++) {
			h_x[i] = other.h_x[i]; h_y[i] = other.h_y[i]; h_z[i] = other.h_z[i];
		}
		if(use_gpu) {
			T *raw_x = nullptr, *raw_y = nullptr, *raw_z = nullptr;
			CUDA_CHECK(cudaMalloc(&raw_x, field_size * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_y, field_size * sizeof(T)));
			CUDA_CHECK(cudaMalloc(&raw_z, field_size * sizeof(T)));
			d_x.reset(raw_x); d_y.reset(raw_y); d_z.reset(raw_z);
			
			CUDA_CHECK(cudaHostRegister(h_x.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_y.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaHostRegister(h_z.get(), field_size * sizeof(T), cudaHostRegisterDefault));
			CUDA_CHECK(cudaMemcpy(d_x.get(), other.d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(d_y.get(), other.d_y.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(d_z.get(), other.d_z.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
	}
	VectorField &operator=(const VectorField &other) {
		if(this == &other) return *this;
		if(!use_gpu) {
			#pragma omp parallel for simd schedule(static)
			for(std::size_t i = 0; i < other.field_size; i++) {
				h_x[i] = other.h_x[i]; h_y[i] = other.h_y[i]; h_z[i] = other.h_z[i];
			}
		} else {
			CUDA_CHECK(cudaMemcpy(d_x.get(), other.d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(d_y.get(), other.d_y.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
			CUDA_CHECK(cudaMemcpy(d_z.get(), other.d_z.get(), field_size * sizeof(T), cudaMemcpyDeviceToDevice));
		}
		return *this;
	}
	void transfer_data_cpu_to_gpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(d_x.get(), h_x.get(), field_size * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_y.get(), h_y.get(), field_size * sizeof(T), cudaMemcpyHostToDevice, stream));
		CUDA_CHECK(cudaMemcpyAsync(d_z.get(), h_z.get(), field_size * sizeof(T), cudaMemcpyHostToDevice, stream));
	}
	void transfer_data_gpu_to_cpu(cudaStream_t stream) noexcept {
		CUDA_CHECK(cudaMemcpyAsync(h_x.get(), d_x.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_y.get(), d_y.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
		CUDA_CHECK(cudaMemcpyAsync(h_z.get(), d_z.get(), field_size * sizeof(T), cudaMemcpyDeviceToHost, stream));
	}
	VectorFieldView<T> get_cpu_view() const noexcept {
		return VectorFieldView<T>(h_x.get(), h_y.get(), h_z.get(), num, r_max, dx, inv_dx, field_size);
	}
	VectorFieldView<T> get_gpu_view() const noexcept {
		return VectorFieldView<T>(d_x.get(), d_y.get(), d_z.get(), num, r_max, dx, inv_dx, field_size);
	}
	VectorField(const Parameters<T> &parameters, const Laser<T> &laser)
		: VectorField(parameters.num, parameters.max_dim_mult * laser.w0, parameters.use_gpu) {}
	VectorField(VectorField &&other) noexcept = default;
	VectorField &operator=(VectorField &&other) noexcept = default;
	~VectorField() = default;
};

struct DataVTK {
	bool use_gpu;
	std::size_t field_size;
	std::unique_ptr<uint32_t[], HostMemoryGeneric<uint32_t>> vtk_scalar, vtk_vector;
	DataVTK(cuda::std::array<int, 3> num, bool use_gpu_new) : use_gpu(use_gpu_new) {
		field_size = num[0] * num[1] * num[2];
		vtk_scalar.reset(new (static_cast<std::align_val_t>(mem_align)) uint32_t[field_size]);
		vtk_vector.reset(new (static_cast<std::align_val_t>(mem_align)) uint32_t[3 * field_size]);
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < field_size; i++)
			vtk_scalar[i] = 0u;
		#pragma omp parallel for simd schedule(static)
		for(std::size_t i = 0; i < 3 * field_size; i++)
			vtk_vector[i] = 0u;
	}
	DataVTK(const Parameters<double> &parameters)
		: DataVTK(parameters.num, parameters.use_gpu) {}
	DataVTK(const Parameters<float> &parameters)
		: DataVTK(parameters.num, parameters.use_gpu) {}
	DataVTK(DataVTK &&other) noexcept = default;
	DataVTK &operator=(DataVTK &&other) noexcept = default;
	~DataVTK() = default;
};

#endif