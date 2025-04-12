# -*- coding: utf-8 -*-
# Author: Mikhail Polyanskiy
# Last modified: 2017-07-08
# Original data: Adachi 1989, https://doi.org/10.1063/1.343580
# Modified by Quinton Mincy 2/25


import numpy as np
import matplotlib.pyplot as plt
π = np.pi

# model parameters
E0   = 1.35    #eV
Δ0   = 1.45-E0 #eV
E1   = 3.10    #eV
Δ1   = 3.25-E1 #eV
E2   = 4.7     #eV
Eg   = 2.05    #eV
A    = 6.57    #eV**1.5
B1   = 4.93
B11  = 10.43   #eV**-0.5
Γ    = 0.10    #eV
C    = 1.49
γ    = 0.094
D    = 60.4
εinf = 1.6

def H(x): #Heviside function
    return 0.5 * (np.sign(x) + 1)

def Epsilon_A(ħω): #E0
    χ0 = ħω/E0
    χso = ħω / (E0+Δ0)
    H0 = H(1-χ0)
    Hso = H(1-χso)
    fχ0 = χ0**-2 * ( 2 -(1+χ0)**0.5 - ((1-χ0)*H0)**0.5 )
    fχso = χso**-2 * ( 2 - (1+χso)**0.5 - ((1-χso)*Hso)**0.5 )
    H0 = H(χ0-1)
    Hso = H(χso-1)
    ε2 = A/(ħω)**2 * ( ((ħω-E0)*H0)**0.5 + 0.5*((ħω-E0-Δ0)*Hso)**0.5)
    ε1 = A*E0**-1.5 * (fχ0+0.5*(E0/(E0+Δ0))**1.5*fχso)
    return ε1 + 1j*ε2
    
def Epsilon_B(ħω): #E1
    # ignoring E1+Δ1 contribution - no data on B2 & B21 in the paper
    χ1 = ħω/E1
    H1 = H(1-χ1)
    ε2 = π*χ1**-2*(B1-B11*((E1-ħω)*H1)**0.5)
    ε2 *= H(ε2) #undocumented trick: ignore negative ε2
    χ1 = (ħω+1j*Γ)/E1
    ε1 = -B1*χ1**-2*np.log(1-χ1**2)
    return ε1.real + 1j*ε2.real

def Epsilon_C(ħω): #E2
    χ2 = ħω/E2
    ε2 = C*χ2*γ / ((1-χ2**2)**2+(χ2*γ)**2)
    ε1 = C*(1-χ2**2) / ((1-χ2**2)**2+(χ2*γ)**2)
    return ε1 + 1j*ε2

def Epsilon_D(ħω): #Eg
    # ignoring ħωq - no data in the paper
    Ech = E1
    χg = Eg/ħω
    χch = ħω/Ech
    Hg = H(1-χg)
    Hch = H(1-χch)
    ε2 = D/ħω**2 * (ħω-Eg)**2 * Hg * Hch
    return 1j*ε2
    
def inp(ev_min, ev_max, npoints, output_file='inp_epsilon.csv'):
    eV = np.logspace(np.log10(ev_min), np.log10(ev_max), npoints)
    μm = 4.13566733e-1*2.99792458/eV
    nm = μm*1000

    εA  = Epsilon_A(eV)
    εB  = Epsilon_B(eV)
    εC  = Epsilon_C(eV)
    εD  = Epsilon_D(eV)    
    ε = εA + εB + εC + εD + εinf
    n = (ε**.5).real
    k = (ε**.5).imag
    α = 4*π*k/μm*1e4 #1/cm

    # Output data to a file
    with open(output_file, 'w') as file:
        for i in range(npoints-1, -1, -1):
            file.write('{:.4e},{:.4e},{:.4e},{:.4e},{:.4e}\n'.format(nm[i], n[i], k[i], ε[i].real, ε[i].imag))

if __name__ == "__main__":
    inp()
