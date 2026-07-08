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

# ---------------------------------------------------------- #

import scripts.sim_init as sim_init
import scripts.programs as programs

c = 137.036

# ---------------------------------------------------------- #

use_gpu = True
use_floats = True

zeta_x_real = 0.707
zeta_x_imag = 0.000
zeta_y_real = 0.000
zeta_y_imag = -0.707

steps = 5000
substeps = 250
max_dim_mult = 3.0

nx = 128
a0 = 1.0
p = 0
m = 0
tau = 10.0
omega = 0.057
w0_mult = 2.0
psi = -0.5 * c * tau
tf = 25.0 * c * tau

# ---------------------------------------------------------- #

if __name__ == "__main__":
    sim_parameters = sim_init.SimParameters(zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag, tf, steps, substeps, nx, a0, p, m, w0_mult, omega, tau, psi, max_dim_mult, use_gpu, use_floats)
    
    programs.run_simulation(sim_parameters)
    
    print("Exa-Beam finished!\a")

# ---------------------------------------------------------- #