# -*- coding: utf-8 -*-
# Author: Mikhail Polyanskiy
# Last modified: 2017-04-02
# Original data: Rakić et al. 1998, https://doi.org/10.1364/AO.37.005271
# Modified by Quinton Mincy 2/25

import numpy as np
import matplotlib.pyplot as plt

# Lorentz-Drude (LD) model parameters
ωp = 9.03  # eV
f0 = 0.760
Γ0 = 0.053  # eV

f1 = 0.024; Γ1 = 0.241; ω1 = 0.415
f2 = 0.010; Γ2 = 0.345; ω2 = 0.830
f3 = 0.071; Γ3 = 0.870; ω3 = 2.969
f4 = 0.601; Γ4 = 2.494; ω4 = 4.304
f5 = 4.384; Γ5 = 2.214; ω5 = 13.32

Ωp = f0**0.5 * ωp  # eV

def LD(ω):
    ε = 1 - Ωp**2 / (ω * (ω + 1j * Γ0))
    ε += f1 * ωp**2 / ((ω1**2 - ω**2) - 1j * ω * Γ1)
    ε += f2 * ωp**2 / ((ω2**2 - ω**2) - 1j * ω * Γ2)
    ε += f3 * ωp**2 / ((ω3**2 - ω**2) - 1j * ω * Γ3)
    ε += f4 * ωp**2 / ((ω4**2 - ω**2) - 1j * ω * Γ4)
    ε += f5 * ωp**2 / ((ω5**2 - ω**2) - 1j * ω * Γ5)
    return ε

def au_model(ev_min, ev_max, npoints, output_file='outLD.csv'):
    eV = np.logspace(np.log10(ev_min), np.log10(ev_max), npoints)
    μm = 4.13566733e-1 * 2.99792458 / eV
    nm = μm*1000
    ε = LD(eV)
    n = (ε**0.5).real
    k = (ε**0.5).imag

    # Output data to a file
    with open(output_file, 'w') as file:
        for i in range(npoints-1, -1, -1):
            file.write('{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}\n'.format(nm[i], n[i], k[i], ε[i].real, ε[i].imag))

if __name__ == "__main__":
    au_model()

