# -*- coding: utf-8 -*-
# Author: Mikhail Polyanskiy
# Last modified: 2017-04-02
# Original data: Rakić et al. 1998, https://doi.org/10.1364/AO.37.005271
# Modified by Quinton Mincy 12/24


import numpy as np
import matplotlib.pyplot as plt
from scipy.special import wofz as w
π = np.pi

# Brendel-Bormann (BB) model parameters
ωp = 9.03  #eV
f0 = 0.770
Γ0 = 0.050 #eV

f1 = 0.054
Γ1 = 0.074 #eV
ω1 = 0.218 #eV
σ1 = 0.742 #eV

f2 = 0.050
Γ2 = 0.035 #eV
ω2 = 2.885 #eV
σ2 = 0.349 #eV

f3 = 0.312
Γ3 = 0.083 #eV
ω3 = 4.069 #eV
σ3 = 0.830 #eV

f4 = 0.719
Γ4 = 0.125 #eV
ω4 = 6.137 #eV
σ4 = 1.246 #eV

f5 = 1.648
Γ5 = 0.179 #eV
ω5 = 27.97 #eV
σ5 = 1.795 #eV

Ωp = f0**.5 * ωp  #eV

def BB(ω):  #ω: eV
    ε = 1-Ωp**2/(ω*(ω+1j*Γ0))

    α = (ω**2+1j*ω*Γ1)**.5
    za = (α-ω1)/(2**.5*σ1)
    zb = (α+ω1)/(2**.5*σ1)
    ε += 1j*π**.5*f1*ωp**2 / (2**1.5*α*σ1) * (w(za)+w(zb)) #χ1
    
    α = (ω**2+1j*ω*Γ2)**.5
    za = (α-ω2)/(2**.5*σ2)
    zb = (α+ω2)/(2**.5*σ2)
    ε += 1j*π**.5*f2*ωp**2 / (2**1.5*α*σ2) * (w(za)+w(zb)) #χ2
    
    α = (ω**2+1j*ω*Γ3)**.5
    za = (α-ω3)/(2**.5*σ3)
    zb = (α+ω3)/(2**.5*σ3)
    ε += 1j*π**.5*f3*ωp**2 / (2**1.5*α*σ3) * (w(za)+w(zb)) #χ3
    
    α = (ω**2+1j*ω*Γ4)**.5
    za = (α-ω4)/(2**.5*σ4)
    zb = (α+ω4)/(2**.5*σ4)
    ε += 1j*π**.5*f4*ωp**2 / (2**1.5*α*σ4) * (w(za)+w(zb)) #χ4
    
    α = (ω**2+1j*ω*Γ5)**.5
    za = (α-ω5)/(2**.5*σ5)
    zb = (α+ω5)/(2**.5*σ5)
    ε += 1j*π**.5*f5*ωp**2 / (2**1.5*α*σ5) * (w(za)+w(zb)) #χ5 
    
    return ε

def BB_model(ev_min, ev_max, npoints, output_file='outBB.csv'):
    eV = np.logspace(np.log10(ev_min), np.log10(ev_max), npoints)
    μm = 4.13566733e-1 * 2.99792458 / eV
    nm = μm*1000
    ε = BB(eV)
    n = (ε**0.5).real
    k = (ε**0.5).imag

    # Output data to a file
    with open(output_file, 'w') as file:
        for i in range(npoints-1, -1, -1):
            file.write('{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}\n'.format(nm[i], n[i], k[i], ε[i].real, ε[i].imag))

if __name__ == "__main__":
    BB_model()
