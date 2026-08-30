# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

from pathlib import Path

MAIN_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MAIN_DIR.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"

class SimParameters():
    def __init__(self, tf, steps, substeps, nx, max_dim_mult, use_gpu, use_floats, output_laser_fields):
        self.tf = tf
        self.nx = nx
        self.steps = steps
        self.use_gpu = use_gpu
        self.substeps = substeps
        self.use_floats = use_floats
        self.max_dim_mult = max_dim_mult
        self.output_directory = OUTPUT_DIR
        self.input_file = INPUT_DIR / "input.txt"
        self.input_laser = INPUT_DIR / "laser.txt"
        self.output_laser_fields = output_laser_fields
        
class Laser():
    def __init__(self, zeta_x, zeta_y, phi, theta, r0, a0, p, m, w0_mult, omega, tau, psi):
        self.p = p
        self.m = m
        self.r0 = r0
        self.a0 = a0
        self.tau = tau
        self.psi = psi
        self.phi = phi
        self.theta = theta
        self.omega = omega
        self.zeta_x = zeta_x
        self.zeta_y = zeta_y
        self.w0_mult = w0_mult