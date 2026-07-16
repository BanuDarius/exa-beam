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

#ifndef CUDA_EXTRA_H
#define CUDA_EXTRA_H

#include <new>
#include <cstdint>

#include <cuda_runtime.h>

constexpr int mem_align = 32;
constexpr int threads_3d_nx = 32;
constexpr int threads_3d_ny = 8;
constexpr int threads_3d_nz = 2;
constexpr int threads_1d_nx = 256;

template <typename T> struct CUDADeviceMemoryAdmin {
	void operator()(T *ptr) noexcept {
		if(ptr) cudaFree(ptr);
	}
};

template <typename T> struct CUDAHostMemoryAdmin {
	void operator()(T *ptr) noexcept {
		if(ptr) { 
			cudaError_t err = cudaHostUnregister(ptr);
			if(err == cudaErrorHostMemoryNotRegistered) cudaGetLastError();
			::operator delete[](ptr, static_cast<std::align_val_t>(mem_align));
		}
	}
};

struct MemoryAdmin {
	void operator()(uint32_t *ptr) noexcept {
		if(ptr) {
			::operator delete[](ptr, static_cast<std::align_val_t>(mem_align));
		}
	}
};

#define CUDA_CHECK(function) \
	do { \
		cudaError_t err = (function); \
		if(err != cudaSuccess) { \
			std::fprintf(stderr, "%s\n", cudaGetErrorString(err)); \
			std::exit(1);\
		} \
	} while(false)

#endif