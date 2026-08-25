// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#include "higuera_cary.hpp"

template<std::floating_point T>
__global__ void higuera_cary_step_kernel(ParticlesView<T> particles_view, __grid_constant__ const DeviceLasers<T> lasers, T t, T dt) {
	int laser_count = lasers.laser_count;
	std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < particles_view.particle_num) {
		cuda::std::array<T, 3> r_vec_global = particles_view.get_position(idx);
		cuda::std::array<T, 3> u_vec_global = particles_view.get_velocity(idx);
		
		EBVectors<T> eb_vec;
		for(int i = 0; i < laser_count; i++) {
			cuda::std::array<T, 3> r_vec_local = lasers[i].pos_global_to_local(r_vec_global);
			EBVectors eb_vec_local = compute_eb(lasers[i], r_vec_local, t);
			eb_vec = EBVectors(
				eb_vec.e + lasers[i].vec_local_to_global(eb_vec_local.e),
				eb_vec.b + lasers[i].vec_local_to_global(eb_vec_local.b)
			);
		}
		
		cuda::std::array<T, 3> beta = hc_beta(eb_vec.b, dt);
		cuda::std::array<T, 3> epsilon = hc_epsilon(eb_vec.e, dt);
		cuda::std::array<T, 3> u_minus = hc_u_minus(u_vec_global, epsilon);
		
		T gamma_minus = comp_gamma(u_minus);
		T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
		
		cuda::std::array<T, 3> t_rot = hc_t_rot(beta, gamma_new);
		T s_factor = hc_s_factor(t_rot);
		cuda::std::array<T, 3> u_prime = hc_u_prime(u_minus, t_rot);
		cuda::std::array<T, 3> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
		
		u_vec_global = u_plus + epsilon;
		T gamma_half = comp_gamma(u_vec_global);
		r_vec_global += u_vec_global * (dt / gamma_half);
		
		particles_view.set_gamma(gamma_half, idx);
		particles_view.set_position(r_vec_global, idx);
		particles_view.set_velocity(u_vec_global, idx);
	}
}

template <std::floating_point T>
void higuera_cary_update_gpu(Particles<T> &particles, const DeviceLasers<T> &lasers, T t, T dt) noexcept {
	std::size_t particle_num = particles.particle_num;
	dim3 threads(threads_1d_nx);
	dim3 blocks((particle_num + threads.x - 1) / threads.x);
	
	ParticlesView<T> particles_view = particles.get_gpu_view();
	higuera_cary_step_kernel<<<blocks, threads>>>(particles_view, lasers, t, dt);
	CUDA_CHECK(cudaGetLastError());
}

template __global__ void higuera_cary_step_kernel<double>(ParticlesView<double> particles_view, __grid_constant__ const DeviceLasers<double> lasers, double t, double dt);
template void higuera_cary_update_gpu<double>(Particles<double> &particles, const DeviceLasers<double> &lasers, double t, double dt) noexcept;

template __global__ void higuera_cary_step_kernel<float>(ParticlesView<float> particles_view, __grid_constant__ const DeviceLasers<float> lasers, float t, float dt);
template void higuera_cary_update_gpu<float>(Particles<float> &particles, const DeviceLasers<float> &lasers, float t, float dt) noexcept;