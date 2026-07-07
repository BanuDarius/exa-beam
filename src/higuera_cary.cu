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

#include "higuera_cary.hpp"

template<std::floating_point T>
__global__ void higuera_cary_step_kernel(ParticlesView<T> particles_view, Laser<T> laser, T t, T dt) {
	std::size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
	if(idx < particles_view.particle_num) {
		cuda::std::array<T, 3> r_vec = particles_view.get_position(idx);
		cuda::std::array<T, 3> u_vec = particles_view.get_velocity(idx);
		T gamma = particles_view.get_gamma(idx);
		
		T half_dt_gamma = T(0.5) * dt / gamma;
		r_vec += u_vec * half_dt_gamma;
		
		EBVectors eb_vec = compute_eb(laser, r_vec, t + T(0.5) * dt);
		
		cuda::std::array<T, 3> beta = hc_beta(eb_vec.b, dt);
		cuda::std::array<T, 3> epsilon = hc_epsilon(eb_vec.e, dt);
		cuda::std::array<T, 3> u_minus = hc_u_minus(u_vec, epsilon);
		
		T gamma_minus = comp_gamma(u_minus);
		T gamma_new = hc_gamma_new(u_minus, beta, gamma_minus);
		
		cuda::std::array<T, 3> t_rot = hc_t_rot(beta, gamma_new);
		T s_factor = hc_s_factor(t_rot);
		cuda::std::array<T, 3> u_prime = hc_u_prime(u_minus, t_rot);
		cuda::std::array<T, 3> u_plus = hc_u_plus(u_minus, u_prime, t_rot, s_factor);
		
		cuda::std::array<T, 3> u_final = u_plus + epsilon;
		gamma = comp_gamma(u_final);
		half_dt_gamma = T(0.5) * dt / gamma;
		r_vec += u_final * half_dt_gamma;

		particles_view.set_position(r_vec, idx);
		particles_view.set_velocity(u_final, idx);
		particles_view.set_gamma(gamma, idx);
	}
}

template <std::floating_point T>
void higuera_cary_update_gpu(Particles<T> &particles, const Laser<T> &laser, T t, T dt) noexcept {
	std::size_t particle_num = particles.particle_num;
	dim3 threads(threads_1d_nx);
	dim3 blocks((particle_num + threads.x - 1) / threads.x);
	
	ParticlesView<T> particles_view = particles.get_gpu_view();
	higuera_cary_step_kernel<<<blocks, threads>>>(particles_view, laser, t, dt);
}

template void higuera_cary_update_gpu<double>(Particles<double> &particles, const Laser<double> &laser, double t, double dt) noexcept;

template void higuera_cary_update_gpu<float>(Particles<float> &particles, const Laser<float> &laser, float t, float dt) noexcept;