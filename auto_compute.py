# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

# ---------------------------------------------------------- #

import numpy as np
import scripts.sim_init as sim_init
import scripts.programs as programs

c = 137.036

# ---------------------------------------------------------- #

use_gpu = True
use_floats = True
output_laser_fields = False

zeta_x_real = 0.707
zeta_x_imag = 0.000
zeta_y_real = 0.000
zeta_y_imag = -0.707

steps = 5000
substeps = 100
max_dim_mult = 3.0

nx = 128
a0 = 0.75
p = 0
m = 1
tau = 4.0
omega = 0.057
w0_mult = 2.0
psi = -0.5 * c * tau
tf = 25.0 * c * tau

phi = np.radians(0.0)
theta = np.radians(0.0)

# ---------------------------------------------------------- #

if __name__ == "__main__":
    sim_parameters = sim_init.SimParameters(tf, steps, substeps, nx, max_dim_mult, use_gpu, use_floats, output_laser_fields)
    
    laser = sim_init.Laser(zeta_x_real, zeta_x_imag, zeta_y_real, zeta_y_imag, phi, theta, a0, p, m, w0_mult, omega, tau, psi)
    
    programs.run_simulation(sim_parameters, laser)
    
    print("Exa-Beam finished!\a")

# ---------------------------------------------------------- #