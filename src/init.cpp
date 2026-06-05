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

#include <cstdlib>
#include <cstring>

#include "init.hpp"
#include "sim_structs.hpp"

template <typename T>
void read_input_file(FILE *in, Parameters<T> &parameters) {
	double r_max;
	int i = 0, num;
	char current[string_size];
	
	while(std::fscanf(in, "%s", current) != EOF) {
		if(!std::strcmp(current, "num"))
			i += std::fscanf(in, "%i", &num);
		else if(!std::strcmp(current, "r_max"))
			i += std::fscanf(in, "%lf", &r_max);
	}
	if(i != input_file_count) {
		std::fprintf(stderr, "INVALID INPUT FILE!\n"); std::exit(1);
	}
	parameters = Parameters(num, T(r_max));
}

template void read_input_file<double>(FILE *in, Parameters<double> &parameters);

template void read_input_file<float>(FILE *in, Parameters<float> &parameters);