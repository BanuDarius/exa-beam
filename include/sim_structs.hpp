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

#include <memory>

template <typename T>
struct Particles {
	int num_x, num_y, num_z;
	std::unique_ptr<T[]> x, y, z, px, py, pz;
	Particles(int nx, int ny, int nz) : num_x(nx), num_y(ny), num_z(nz) {
		int total = num_x * num_y * num_z;
		x = std::unique_ptr<T[]>(new T[total]);
		y = std::unique_ptr<T[]>(new T[total]);
		z = std::unique_ptr<T[]>(new T[total]);
		px = std::unique_ptr<T[]>(new T[total]);
		py = std::unique_ptr<T[]>(new T[total]);
		pz = std::unique_ptr<T[]>(new T[total]);
		#pragma omp parallel for simd schedule(static)
		for(int i = 0; i < total; i++) {
			x[i] = T(0.0); y[i] = T(0.0); z[i] = T(0.0);
			px[i] = T(0.0); py[i] = T(0.0); pz[i] = T(0.0);
		}
	}
};

#endif