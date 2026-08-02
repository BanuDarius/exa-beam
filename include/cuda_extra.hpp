// Copyright (c) 2026 Banu Darius-Matei
// SPDX-License-Identifier: MIT

#ifndef CUDA_EXTRA_H
#define CUDA_EXTRA_H

#include <new>
#include <cstdint>
#include <concepts>

#include <cuda_runtime.h>

constexpr int mem_align = 32;
constexpr int threads_3d_nx = 32;
constexpr int threads_3d_ny = 8;
constexpr int threads_3d_nz = 2;
constexpr int threads_1d_nx = 256;

template <std::floating_point T> struct DeviceMemory {
	void operator()(T *ptr) noexcept {
		if(ptr) cudaFree(ptr);
	}
};

template <typename T> struct DeviceMemoryGeneric {
	void operator()(T *ptr) noexcept {
		if(ptr) cudaFree(ptr);
	}
};

template <std::floating_point T> struct HostMemory {
	void operator()(T *ptr) noexcept {
		if(ptr) { 
			cudaError_t err = cudaHostUnregister(ptr);
			if(err == cudaErrorHostMemoryNotRegistered) cudaGetLastError();
			::operator delete[](ptr, static_cast<std::align_val_t>(mem_align));
		}
	}
};

template <typename T> struct HostMemoryGeneric {
	void operator()(T *ptr) noexcept {
		if(ptr) { 
			cudaError_t err = cudaHostUnregister(ptr);
			if(err == cudaErrorHostMemoryNotRegistered) cudaGetLastError();
			::operator delete[](ptr, static_cast<std::align_val_t>(mem_align));
		}
	}
};

#define CUDA_CHECK(function) \
	do { \
		cudaError_t err = (function); \
		if(err != cudaSuccess) { \
			std::fprintf(stderr, "%s\n", cudaGetErrorString(err)); \
			std::exit(1); \
		} \
	} while(false)
#endif