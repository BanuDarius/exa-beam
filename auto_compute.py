# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

import numpy as np
import scripts.sim_init as sim_init
import scripts.programs as programs
import scripts.render as render

c = 137.036

use_gpu = True
use_floats = True
output_laser_fields = False

nx = 128
steps = 12000
substeps = 50
max_dim_mult = [ 3.0, 3.0, 3.0 ]

p = 0
m = 0
a0 = 1.5
tau = 8.0
omega = 0.057
w0_mult = 4.0
offset = max_dim_mult[2] * w0_mult
psi = - 12.0 * np.pi * offset
tf = - 2.0 * psi / omega

phi = np.radians(0.0)
theta = np.radians(0.0)

r0 = [ 0.0, 0.0, 0.0 ]
zeta_x = [ 0.707, 0.000 ]
zeta_y = [ 0.000, -0.707 ]

if __name__ == "__main__":
    sim_parameters = sim_init.SimParameters(tf, steps, substeps, nx, max_dim_mult, use_gpu, use_floats, output_laser_fields)
    
    lasers = []
    lasers.append(sim_init.Laser(zeta_x, zeta_y, phi, theta, r0, a0, p, m, w0_mult, omega, tau, psi))
    lasers.append(sim_init.Laser(zeta_x, zeta_y, phi, np.radians(90.0), r0, a0, p, m, w0_mult, omega, tau, psi))
    
    programs.run_simulation(sim_parameters, lasers)
    
    render.render_paraview()
    
    print("Exa-Beam finished!\a")