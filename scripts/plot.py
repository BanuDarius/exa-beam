# Copyright (c) 2026 Banu Darius-Matei
# SPDX-License-Identifier: MIT

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

def plot_divergence():
    filename = "output/diagnostics.bin"
    output = "output-image/divergence.png"
    data = np.fromfile(filename, dtype=np.float32).reshape(-1, 2)
    divergence = data[:, 1]
    
    plt.figure()
    plt.plot(divergence)
    plt.xlabel("Step / substep")
    plt.ylabel(r"$\sqrt{\sum\left(\nabla \cdot \mathbf{E}\right)^2}$")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()

def plot_total_energy():
    filename = "output/diagnostics.bin"
    output = "output-image/total_energy.png"
    data = np.fromfile(filename, dtype=np.float32).reshape(-1, 2)
    energy = data[:, 0]
    
    plt.figure()
    plt.plot(energy)
    plt.xlabel("Step / substep")
    plt.ylabel(r"$\int w_{em} dV$")
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
