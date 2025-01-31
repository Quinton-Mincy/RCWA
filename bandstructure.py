#Author(s): Quinton Mincy
#Last Modified: 01/31/24
#Code Built Arround Alex Song's Examples for his Inkstone library: https://github.com/alexysong/inkstone

import numpy as np
from numpy import linalg as la

from matplotlib import pyplot as plt
from matplotlib import cm
# import au_LD_model
import math
from inkstone import Inkstone
from tqdm import tqdm
import csv
from typing import List, Tuple

#multipliers
tera = 1e12
#program parameters
c    = 299792458
c_nm = c * 1e9  # Convert to nanometers per second (nm/s)

#should be multiples of 90
num_points = 30

latc = 0.5

a1 = (latc,0)
a2 = ((1/2)*latc,latc*math.sqrt(3)/2)

# Define hexagon parameters
center = (0, 0)  # Center of the hexagon
radius = 0.216/2    # circumradius (microns)

# latc = 1
# a1=(1,0)
# a2=(0,1)


#find reciprocal lattice vectors
def recipro(a1, a2):
    """
    given two lattice vectors, give two reciprocal lattice vectors
    If one of the lattice vectors is zero, then the returned corresponding reciprocal lattice vector is float('inf').

    Parameters
    ----------
    a1  :   tuple[float, float]
    a2  :   tuple[float, float]

    Returns
    -------
    b1  :   tuple[float, float]
    b2  :   tuple[float, float]

    """

    a1, a2 = [np.array(a) for a in [a1, a2]]
    a1n, a2n = [la.norm(a) for a in [a1, a2]]

    if a1n == 0.:
        if a2n == 0.:
            raise Exception("The two lattice vectors can't be both zero vectors.")
        else:
            b1 = (float('inf'), float('inf'))
            b2 = tuple(2 * np.pi / (a2n ** 2) * a2)
    else:
        if a2n == 0.:
            b1 = tuple(2 * np.pi / (a1n ** 2) * a1)
            b2 = (float('inf'), float('inf'))
        else:
            ar = np.abs(np.cross(a1, a2))  # area
            coef = 2 * np.pi / ar

            b1 = np.array([coef * a2[1], -coef * a2[0]])
            b2 = np.array([-coef * a1[1], coef * a1[0]])

    return b1,b2

#angle between reciprocal lattice vectors (radians)
def find_phi(b1,b2):
    dot = np.dot(b1,b2)
    mag_b1 = np.linalg.norm(b1)
    mag_b2 = np.linalg.norm(b2)

    phi = np.arccos(dot/(mag_b1 * mag_b2))

    return phi
def k_corner(b1,b2):
    corner = b1 + b2
    corner = corner / np.linalg.norm(corner)
    return corner

def freq2lambda(a,freq):
    #wavelegnth = lattice_constant/norm_freq
    return a*(freq)**-1

#projection of a on b
def proj(a,b):
    b_b = np.dot(b,b)
    if(b_b) == 0:
        return 0
    else:
        proj_a_on_b = ( np.dot(a,b) / b_b ) * b
    return proj_a_on_b

#incident excitation source
def wavevector(theta,phi,ω):
    #kx,ky
    #k = np.array( [(ω/c) * math.sin(theta)*math.cos(phi) , (ω/c) * math.sin(theta)*math.sin(phi) ] )
    
    kx =  ω * np.cos(np.pi/2 - theta) * np.cos(phi)
    ky =  ω * np.cos(np.pi/2 - theta) * np.sin(phi)
    k = np.array([kx,ky])
  
    return k

# Function to generate hexagon vertices
def generate_hexagon(center,radius):
    cx, cy = center
    vertices = [
        (cx + radius * np.cos(2 * np.pi * i / 6), cy + radius * np.sin(2 * np.pi * i / 6)) for i in range(6)]
    return vertices

def init_inkstone(epsilon,geometry):
    s = Inkstone()
    s.lattice = ((a1,a2))
    s.num_g = 50

    s.AddMaterial(name='Au', epsilon=epsilon)
    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=1.0, material_background='Au')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)
    # s.AddPatternDisk(layer='slab', pattern_name='disk', material='vacuum', radius=0.216)
    s.AddPatternPolygon(layer = 'slab',material = 'Au',vertices = geometry,pattern_name = 'Hexagon')

    return s

def plot_spectrum(epsilon,frequency,wavelength):
    #initiate inkstone object
    hexagon_vertices = generate_hexagon(center, radius)
    s = init_inkstone(epsilon,hexagon_vertices)
    
    #reciprocal space calculations
    b1,b2 =  recipro(a1,a2)
    corner = k_corner(b1,b2)
    phi_bis = find_phi(b1,b2) / 2
    phi_bis_deg = phi_bis * (180/math.pi)

    #sweep through incident angle
    theta_values1 = np.linspace(90,0,num_points)
    angles_deg = np.concatenate((theta_values1, theta_values1[::-1]))
    angles_rad = [math.radians(angle) for angle in angles_deg]

    #x and z axis arrays
    kx_values = []
    transmission_values = []

    phis = [phi_bis_deg,90]
    phis_rad = [phi_bis,(math.pi)/2]

    vec_ = [corner, b1]
    freq_normalized = (frequency * (500e-9)) / c

    with tqdm(total=len(freq_normalized), desc="Analyzing Frequencies", dynamic_ncols=True) as pbar:
        for i, nu in enumerate(freq_normalized):
            pbar.set_description(f"ν: {frequency[i]/tera:.2f} THz, λ: {wavelength[i]:.2f} nm")  # Update text
            pbar.update(1)  # Increment progress bar
            # Update material properties and frequency
            s.SetMaterial(name='Au', epsi=epsilon)
            s.SetFrequency(nu)
            flux_in = []
            flux_out = []

            proj_mags = []
            j = 0
            n = -1
            for i, (theta_d, theta_r) in enumerate(zip(angles_deg, angles_rad)):
                #transmission calculations
                s.SetExcitation(theta=theta_d, phi=phis[j], s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
                flux_in.append(s.GetPowerFlux('in'))
                flux_out.append(s.GetPowerFlux('out'))
                #projections
                wv =wavevector(theta_r,phis_rad[j],nu) 
                projection = [proj(wv,vec_[j])]
                #find kx
                proj_mag = np.linalg.norm(projection) 
                #make k-corner values negative
                proj_mag = proj_mag * n
                proj_mags.append(proj_mag)
                if(i == num_points - 1):
                    j = 1
                    n = 1

            incident = np.array([a[0] for a in flux_in])
            reflection = -np.array([a[1] for a in flux_in]) / incident
            transmission = np.array([a[0] for a in flux_out]) / incident

            transmission_values.append(transmission)
            
            kx_values.append(proj_mags)

    #Transmision vs wavelength
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Transmission (%)')
    # plt.title('Transmission vs Wavelength for Various Theta Values')
    # plt.legend(loc='best')  # Automatically place legend
    # plt.grid(True)

    # Prepare data for contour plot
    kx_values = np.array(kx_values)
    # kx_values *= ( (2*math.pi) / 500e-9)
    transmission_values = np.array(transmission_values)

    num_y, num_x = kx_values.shape  # Get dimensions

    Y = frequency.reshape(-1, 1)
    # Expand Y across columns to match X (broadcasting)
    Y = np.tile(Y, (1, num_x))

    # image show transmission
    plt.figure(figsize=(10, 6))
    plt.imshow(transmission_values,cmap = 'viridis', origin='lower')


    # #Contour map
    plt.figure(figsize=(10, 6))
    plt.contourf(kx_values, Y, transmission_values, levels=100, cmap='viridis')  # Colormap

    plt.colorbar(label='Transmission')
    plt.xlabel('$k_x$')
    plt.ylabel('Frequency (THz)')
    plt.title('Transmission Intensity Contour ($k_x$ vs. Frequency)')

def eps():
    # points = extract(data)
    #wavelength
    # lam = points[0]
    # #index of frefraction
    # n   = points[1]
    # #extinction coeeficient
    # k   = points[2]
    # #dielectric constant
    # ε_real   = points[3]
    # ε_imag   = points[4]
    #375e12-420e12 THz ~ 714-800 nm
    #
    # Define the Near-Infrared (NIR) frequency range in Hz
    freq_nir = np.linspace(400e12,290e12, num_points*2)
    # Convert frequency to wavelength in nanometers
    wavelength_nir = c_nm / freq_nir  # λ = c / f (in nm)
    # epsilon = [complex(real, imag) for real, imag in zip(ε_real, ε_imag)]  
    plot_spectrum(9.61,freq_nir,wavelength_nir)

    plt.show()

if __name__ =='__main__':
    # au_LD_model.au_model()
    # data = np.genfromtxt('out.csv', delimiter=',')
    eps()
