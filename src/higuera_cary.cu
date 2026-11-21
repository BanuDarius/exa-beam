// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include "higuera_cary.hpp"

template <std::floating_point T>
__global__ void hc_update_gpu_kernel(ParticlesView<T> particles, const VectorFieldView<T> e, const VectorFieldView<T> b, T dt) {
	std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < particles.particle_num) {
		Vector3<T> r_vec = particles.get_position(idx);
		Vector3<T> u_vec = particles.get_velocity(idx);
		
		EBVector3<T> eb_vec = cic_gather(r_vec, e, b);
		Vector3<T> e_vec = eb_vec[0], b_vec = eb_vec[1];
		
		Vector3<T> beta = hc_beta(b_vec, dt);
		Vector3<T> epsilon = hc_epsilon(e_vec, dt);
		Vector3<T> u_minus = hc_u_minus(u_vec, epsilon);
		
		T gamma_minus = hc_gamma(u_minus);
		T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
		
		Vector3<T> t_rot = hc_t_rot(beta, gamma_new);
		T s_factor = hc_s_factor(t_rot);
		Vector3<T> u_prime = hc_u_prime(u_minus, t_rot);
		Vector3<T> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
		
		u_vec = u_plus + epsilon;
		T gamma_half = hc_gamma(u_vec);
		r_vec += u_vec * (dt / gamma_half);
		
		particles.set_gamma(gamma_half, idx);
		particles.set_position(r_vec, idx);
		particles.set_velocity(u_vec, idx);
	}
}

template <std::floating_point T>
void hc_update_gpu(Particles<T> &particles_data, const VectorField<T> &e_data, const VectorField<T> &b_data, T dt) noexcept {
	std::size_t particle_num = particles_data.particle_num;
	dim3 threads(threads_1d_nx);
	dim3 blocks((particle_num + threads.x - 1) / threads.x);
	
	ParticlesView<T> particles= particles_data.get_gpu_view();
	VectorFieldView<T> e = e_data.get_gpu_view(), b = b_data.get_gpu_view();
	hc_update_gpu_kernel<<<blocks, threads>>>(particles, e, b, dt);
	CUDA_CHECK(cudaGetLastError());
}

template __global__ void hc_update_gpu_kernel<double>(ParticlesView<double> particles, const VectorFieldView<double> e, const VectorFieldView<double> b, double dt);
template void hc_update_gpu<double>(Particles<double> &particles_data, const VectorField<double> &e_data, const VectorField<double> &b_data, double dt) noexcept;

template __global__ void hc_update_gpu_kernel<float>(ParticlesView<float> particles, const VectorFieldView<float> e, const VectorFieldView<float> b, float dt);
template void hc_update_gpu<float>(Particles<float> &particles_data, const VectorField<float> &e_data, const VectorField<float> &b_data, float dt) noexcept;
