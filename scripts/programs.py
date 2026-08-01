# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

# ---------------------------------------------------------- #

import subprocess
from pathlib import Path

# ---------------------------------------------------------- #

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------- #

def output_init_files(sim_parameters, laser):
    input_file = sim_parameters.input_file
    laser_file = sim_parameters.input_laser
    
    with open(input_file, "w") as file:
        file.write(f"tf {sim_parameters.tf}\n")
        file.write(f"nx {sim_parameters.nx}\n")
        file.write(f"steps {sim_parameters.steps}\n")
        file.write(f"substeps {sim_parameters.substeps}\n")
        file.write(f"use_gpu {int(sim_parameters.use_gpu)}\n")
        file.write(f"max_dim_mult {sim_parameters.max_dim_mult}\n")
        file.write(f"output_laser_fields {int(sim_parameters.output_laser_fields)}\n")
    
    with open(laser_file, "w") as file:
        file.write(f"r0_x 0.0\n")
        file.write(f"r0_y 0.0\n")
        file.write(f"r0_z 0.0\n")
        file.write(f"p {laser.p}\n")
        file.write(f"m {laser.m}\n")
        file.write(f"a0 {laser.a0}\n")
        file.write(f"tau {laser.tau}\n")
        file.write(f"psi {laser.psi}\n")
        file.write(f"phi {laser.phi}\n")
        file.write(f"theta {laser.theta}\n")
        file.write(f"omega {laser.omega}\n")
        file.write(f"w0_mult {laser.w0_mult}\n")
        file.write(f"zeta_x_real {laser.zeta_x_real}\n")
        file.write(f"zeta_x_imag {laser.zeta_x_imag}\n")
        file.write(f"zeta_y_real {laser.zeta_y_real}\n")
        file.write(f"zeta_y_imag {laser.zeta_y_imag}\n")
        
# ---------------------------------------------------------- #

def run_simulation(sim_parameters, laser):
    output_init_files(sim_parameters, laser)
    
    if(sim_parameters.use_floats == False):
        arguments = [BIN_DIR / "exa_beam", sim_parameters.input_file, sim_parameters.input_laser, sim_parameters.output_directory]
    else:
        arguments = [BIN_DIR / "exa_beam", "--float", sim_parameters.input_file, sim_parameters.input_laser, sim_parameters.output_directory]
    
    subprocess.run(arguments, text=True)
    
# ---------------------------------------------------------- #