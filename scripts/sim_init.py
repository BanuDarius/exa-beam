'''MIT License

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
SOFTWARE.'''

from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------- #

class SimParameters():
    def __init__(self, zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag, tf, steps, substeps, nx, a0, p, m, w0_mult, omega, tau, psi, max_dim_mult, use_gpu, use_floats):
        self.p = p
        self.m = m
        self.a0 = a0
        self.tf = tf
        self.nx = nx
        self.tau = tau
        self.psi = psi
        self.steps = steps
        self.omega = omega
        self.use_gpu = use_gpu
        self.w0_mult = w0_mult
        self.substeps = substeps
        self.use_floats = use_floats
        self.zeta_x_real = zeta_x_real
        self.zeta_x_imag = zeta_x_imag
        self.zeta_y_real = zeta_y_real
        self.zeta_y_imag = zeta_y_imag
        self.max_dim_mult = max_dim_mult
        self.input_file = INPUT_DIR / "input.txt"
        self.output_directory = OUTPUT_DIR
        
# ---------------------------------------------------------- #