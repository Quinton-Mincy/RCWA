#Author(s): Quinton Mincy
#Last Modified: 01/26/24
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


#program parameters
c    = 299792458
#should be multiples of 90
num_points = 3
latc = 0.5
# a1 = (-(1/2)*latc,latc*math.sqrt(3)/2)
# a2 = ((1/2)*latc,latc*math.sqrt(3)/2)
# a1 = (latc,0)
# a2 = (-(1/2)*latc,latc*math.sqrt(3)/2)

a1 = (latc,0)
a2 = ((1/2)*latc,latc*math.sqrt(3)/2)

# Define hexagon parameters
center = (0, 0)  # Center of the hexagon
radius = 0.216/2    # circumradius (microns)

# latc = 1
# a1=(1,0)
# a2=(0,1)


#find reciprocal lattice vectors
def recipro(a1, a2,k_point = False):
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

    # print(b1,b2)
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
    print(vertices)
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

def plot_spectrum(epsilon,frequency):
    
    hexagon_vertices = generate_hexagon(center, radius)
    s = init_inkstone(epsilon,hexagon_vertices)
    
    b1,b2 =  recipro(a1,a2)
    print(b1,b2)
    corner = k_corner(b1,b2)

    phi_bis = find_phi(b1,b2) / 4

    phi_bis_deg = phi_bis * (180/math.pi)



    wavelengths = freq2lambda(1, frequency) * 1000  # Convert to nm
    
    #set figure size
    plt.figure(figsize=(10, 6))

    #sweep through incident angle
    theta_values = np.linspace(90,0,num_points)
    theta_values = np.concatenate((theta_values, theta_values))
    theta_values_rad = [math.radians(theta) for theta in theta_values]

    transmission_values = []
    kx_values = []

    phis = [phi_bis_deg,90]
    phis_rad = [phi_bis,(math.pi)/2]
    j = 0
    k = 0

    

    # Create increasing sequence
    
    freq_rev = frequency[::-1]

    freqs = [frequency,freq_rev]
    
    # Concatenate both sequences
    iters = freqs + freq_rev


    #sweep through incident angle
    theta_values1 = np.linspace(90,0,num_points)
    theta_values_rad = [math.radians(theta) for theta in theta_values1]

    theta_values1_rev = np.linspace(0,90,num_points)
    theta_values1_rev_rad = [math.radians(theta) for theta in theta_values1_rev]

    theta_deg = [theta_values1,theta_values1_rev]

    for i, nu in enumerate(frequency):
        # Update material properties and frequency
        s.SetMaterial(name='Au', epsi=epsilon)
        s.SetFrequency(nu)
        flux_in = []
        flux_out = []
        j = 0
        for theta in tqdm(theta_deg[j]):
            s.SetExcitation(theta=theta, phi=phis[j], s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
            flux_in.append(s.GetPowerFlux('in'))
            flux_out.append(s.GetPowerFlux('out'))
            if theta == 0:
                j = 1
        incident = np.array([a[0] for a in flux_in])
     
        reflection = -np.array([a[1] for a in flux_in]) / incident
        transmission = np.array([a[0] for a in flux_out]) / incident

        transmission_values.append(transmission)

                
    # for theta in tqdm(theta_values):

    #     s.SetExcitation(theta=theta, phi=phis[j], s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
    #     flux_in = []
    #     flux_out = []

    #     for i, nu in enumerate(freqs[j]):
        
            
    #         # Update material properties and frequency
    #         s.SetMaterial(name='Au', epsi=epsilon)
    #         s.SetFrequency(nu)
    #         flux_in.append(s.GetPowerFlux('in'))
    #         flux_out.append(s.GetPowerFlux('out'))

    #     incident = np.array([a[0] for a in flux_in])
     
    #     reflection = -np.array([a[1] for a in flux_in]) / incident
    #     transmission = np.array([a[0] for a in flux_out]) / incident

    #     transmission_values.append(transmission)
    #     if(k == num_points - 1):
    #         j = 1
    #         s = init_inkstone(epsilon,hexagon_vertices)
    #     k = k+1
        
    j = 0
    k = 0
    c = -1
    vec_ = [corner, b1]
    for j in range(len(freqs)):
        for nu in freqs[j]:
            wavevectors = [wavevector(theta,phis_rad[j],nu) for theta in theta_values_rad]
            projections = [proj(wv,vec_[j]) for wv in wavevectors]

            # wavevectors = [wavevector(theta,phis_rad[j+1],nu) for theta in theta_values_rad]
            # projections.append
            proj_mags = [np.linalg.norm(proj) for proj in  projections]
            proj_mags = [mag*c for mag in proj_mags]
            kx_values.append(proj_mags)
            if(k == num_points - 1):
                j = 1
                c = 1
            k = k+1

    
    
    #Transmision vs wavelength
    # plt.xlabel('Wavelength (nm)')
    # plt.ylabel('Transmission (%)')
    # plt.title('Transmission vs Wavelength for Various Theta Values')
    # plt.legend(loc='best')  # Automatically place legend
    # plt.grid(True)

    # Prepare data for contour plot
    kx_values = np.array(kx_values)
    transmission_values = np.array(transmission_values)
    transmission_values_2d = transmission_values
    frequency_2d = np.tile(freqs[0][:, np.newaxis], (1, kx_values.shape[1])) 


    #image show transmission
    plt.figure(figsize=(10, 6))
    plt.imshow(transmission_values_2d,cmap = 'viridis')


    #Contour map
    plt.figure(figsize=(10, 6))
    plt.contourf(kx_values, frequency_2d, transmission_values_2d, levels=100, cmap='viridis')  # Colormap
    plt.colorbar(label='Transmission')
    plt.xlabel('$k_x$')
    plt.ylabel('Wavelength')
    plt.title('Transmission Intensity Contour ($k_x$ vs. Wavelength)')

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
       
    frequency = np.linspace(.25,.6,num_points)
    
    # epsilon = [complex(real, imag) for real, imag in zip(ε_real, ε_imag)]  
    plot_spectrum(9.61,frequency)
 

    plt.show()

if __name__ =='__main__':
    # au_LD_model.au_model()
    # data = np.genfromtxt('out.csv', delimiter=',')
    eps()
