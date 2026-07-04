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

import subprocess
from pathlib import Path

# ---------------------------------------------------------- #

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parent
BIN_DIR = PROJECT_ROOT / "bin"
OUTPUT_DIR = PROJECT_ROOT / "output"

# ---------------------------------------------------------- #

def output_init_file(sim_parameters):
    input_file = sim_parameters.input_file
    
    with open(input_file, "w") as file:
        file.write(f"p {sim_parameters.p}\n")
        file.write(f"m {sim_parameters.m}\n")
        file.write(f"a0 {sim_parameters.a0}\n")
        file.write(f"tf {sim_parameters.tf}\n")
        file.write(f"nx {sim_parameters.nx}\n")
        file.write(f"tau {sim_parameters.tau}\n")
        file.write(f"psi {sim_parameters.psi}\n")
        file.write(f"steps {sim_parameters.steps}\n")
        file.write(f"omega {sim_parameters.omega}\n")
        file.write(f"w0_mult {sim_parameters.w0_mult}\n")
        file.write(f"substeps {sim_parameters.substeps}\n")
        file.write(f"max_dim_mult {sim_parameters.max_dim_mult}\n")
        file.write(f"zeta_x_real {sim_parameters.zeta_x_real}\n")
        file.write(f"zeta_x_imag {sim_parameters.zeta_x_imag}\n")
        file.write(f"zeta_y_real {sim_parameters.zeta_y_real}\n")
        file.write(f"zeta_y_imag {sim_parameters.zeta_y_imag}\n")
        
# ---------------------------------------------------------- #

def run_simulation(sim_parameters):
    output_init_file(sim_parameters)
    
    if(sim_parameters.use_floats == False):
        arguments = [BIN_DIR / "exa_beam", sim_parameters.input_file, sim_parameters.output_directory]
    else:
        arguments = [BIN_DIR / "exa_beam", "--float", sim_parameters.input_file, sim_parameters.output_directory]
    
    subprocess.run(arguments, text=True)
    
# ---------------------------------------------------------- #