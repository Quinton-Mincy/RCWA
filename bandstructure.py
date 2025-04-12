#Author(s): Quinton Mincy
#Last Modified: 02/26/24
#Code Built Arround Alex Song's Examples for his Inkstone library: https://github.com/alexysong/inkstone


import numpy as np
from numpy import linalg as la

from matplotlib import pyplot as plt
from matplotlib import cm
import au_LD_model
# import au_BB_model
import inp
import math
from inkstone import Inkstone
from tqdm import tqdm
import csv



#constants
c    = 299792458
c_nm = c * 1e9  # Convert to nanometers per second (nm/s)

pi   = np.pi
#multipliers
tera = 1e12
#should be multiples of 90
NUM_POINTS = 90
NUM_G = 10 0
#program parameters
# latc = 0.3
# latc_nm = 300e-9
# #microns
# thickness = 0.02

thickness = 1
latc = 0.5
latc_nm = 500e-9

a1 = (latc,0)
a2 = ((1/2)*latc,latc*math.sqrt(3)/2)

# a1 = (1,0)
# a2 = (0,1)
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
            b2 = tuple(2 * pi / (a2n ** 2) * a2)
    else:
        if a2n == 0.:
            b1 = tuple(2 * pi / (a1n ** 2) * a1)
            b2 = (float('inf'), float('inf'))
        else:
            ar = np.abs(np.cross(a1, a2))  # area
            coef = 2 * pi / ar

            b1 = (coef * a2[1], -coef * a2[0])
            b2 = (-coef * a1[1], coef * a1[0])

    return np.array([b1, b2])


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
    kx =  ω * np.cos(np.pi/2 - theta) * np.cos(phi)
    ky =  ω * np.cos(np.pi/2 - theta) * np.sin(phi)
    k = np.array([kx,ky])
  
    return k

def init_inkstone(epsilon,geometry):
    s = Inkstone()
    s.lattice = ((a1,a2))
    s.num_g = NUM_G
    

    s.AddMaterial(name='InP', epsilon=epsilon.real)
    
    s.AddLayer(name='in', thickness=0, material_background='InP')
    s.AddLayer(name='slab', thickness=thickness, material_background='vacuum')
    s.AddLayer(name='out', thickness=0, material_background='vacuum')

    s.AddPatternPolygon(layer="slab", material="InP", pattern_name="poly1",
        vertices=geometry)
    return s

def RCWA(epsilon,frequency,wavelength):
    frequency_norm = normalize_frequency(frequency,(latc_nm))

    # centers=[(0,0),(latc,0),(latc/2,latc*np.sqrt(3)/2 )]
    # geometry = init_geometry(latc/2,centers)
    center = [0,0]
    radius = 0.216/2
    # center = [latc/2,latc*np.sqrt(3)/2]
    geometry = generate_hexagon(center,radius )
    s = init_inkstone(epsilon[0].real,geometry)

    #sweep through incident angle
    # theta_values = np.linspace(0,90,NUM_POINTS)
    # theta_values_rad = [math.radians(the) for the in theta_values]

    theta_values = np.linspace(90,0,NUM_POINTS)
    theta_values = np.concatenate((theta_values, theta_values[::-1]))
    theta_values_rad = [math.radians(angle) for angle in theta_values]

    phis = np.array([60,90])
    phis_rad = np.radians(phis)

    transmission_values = []
    kx_values = []

    K1 = (2*pi/latc)*np.array([1/3, 1/math.sqrt(3)])
    M1 = (2*pi/latc)*np.array([0,1/math.sqrt(3)])
    bz1 = [K1,M1]
    K2 = (2*pi/latc)*np.array([1/3, -1/math.sqrt(3)])
    M2 = (2*pi/latc)*np.array([0,-1/math.sqrt(3)])
    bz2 = [K2,M2]

    #for transmission spectra
    fig, ax = plt.subplots(figsize=(10, 6))

    with tqdm(total=len(frequency_norm), desc="Analyzing Frequencies", dynamic_ncols=True) as pbar:
        for i, nu in enumerate(frequency_norm):
            pbar.set_description(f"ν: {frequency[i]/tera:.2f} THz, λ: {wavelength[i]:.2f} nm, ε:{epsilon[i]} ")  # Update text
            pbar.update(1)  # Increment progress bar
            s.SetMaterial(name='InP', epsi=epsilon[i].real)
            s.SetFrequency(nu)
            flux_in = []
            flux_out = []

            proj_mags = []

            k = 0
            n = -1
            for j, (theta_d, theta_r) in enumerate(zip(theta_values, theta_values_rad)):
        
                wv = wavevector(theta_r,phis_rad[k],nu)
                k_recps =  recipro(a1,a2)
                b1 = np.linalg.norm(k_recps[0])

                projection = proj(wv,bz1[k])
                proj_mag = np.linalg.norm(projection) * n
                proj_mags.append(proj_mag)
            
                s.SetExcitation(theta=theta_d, phi=phis[k], s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
                flux_in.append(s.GetPowerFlux('in'))
                flux_out.append(s.GetPowerFlux('out'))
                if(j == NUM_POINTS - 1):
        
                    k = 1
                    n = 1

            incident = np.array([a[0] for a in flux_in])
            # reflection = -np.array([a[1] for a in flux_in]) / incident
            transmission = np.array([a[0] for a in flux_out]) / incident
            
            transmission_values.append(transmission)
            kx_values.append(proj_mags)


    # Prepare data for contour plot
    kx_values = np.array(kx_values)
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


def freq_to_ev(freq):
    h = 6.626e-34
    unit = 1.602e-19
    e = (h*freq)/unit
    return e

def freq2lambda(freq):
    #wavelegnth = lattice_constant/norm_freq
    return c/freq


def extract(data):
    points = np.array((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]))
    return points

def normalize_frequency(freq,a):
    return freq*(a/c)

def parametric_circle(theta, center, radius):
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    return np.array(list(zip(x, y)))  # Convert to array of (x, y) tuples

def init_geometry(radius,centers):
    theta1 = np.linspace(np.pi/3,0,NUM_POINTS,endpoint=False)
    theta2 = np.linspace(np.pi,2*np.pi/3, NUM_POINTS,endpoint=False)
    theta3 = np.linspace(5*np.pi/3,4*np.pi/3,NUM_POINTS,endpoint=False)
    thetas = [theta1,theta2,theta3]
    sides = []
    
    for theta,center in zip(thetas,centers):
        sides.append(parametric_circle(theta,center,radius))

    geometry = np.concatenate((sides[0],sides[1],sides[2]))
    return geometry

def generate_hexagon(center,radius):
    cx, cy = center
    vertices = [
        (cx + radius * np.cos(2 * np.pi * i / 6), cy + radius * np.sin(2 * np.pi * i / 6)) for i in range(6)]
    return vertices

def plot_geometry(geometry):
    """
    Plots the points generated by init_geometry using Matplotlib.
    
    Parameters:
        geometry (np.ndarray): Array of (x, y) tuples representing the shape.
    """
    # Extract x and y coordinates
    x, y = geometry[:, 0], geometry[:, 1]

    # Plot the points
    plt.figure(figsize=(6, 6))
    plt.plot(x, y, 'bo-', markersize=3, linewidth=1, label="Triangle Edges")  # Blue markers and lines
    plt.scatter(x, y, color='red', s=10, label="Vertices")  # Red dots for individual points
    
    # Labels and grid
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Triangle Geometry Plot")
    plt.legend()
    plt.grid(True)

    # Equal aspect ratio for correct visualization
    plt.axis("equal")

    # Show the plot
    plt.show()

def eps(data,freq_range):
    points = extract(data)
    #dielectric constant
    ε_real   = points[3]
    ε_imag   = points[4]
    epsilon = np.array([complex(real, imag) for real, imag in zip(ε_real, ε_imag)])
    # epsilon = epsilon[::-1]
    #initialize frequency array
    freq_params = np.linspace(freq_range[0],freq_range[1], NUM_POINTS)
    wavelength = c_nm / freq_params  # λ = c / f (in nm)
    wavelength = wavelength
    #RCWA
    RCWA(epsilon,freq_params,wavelength)
    plt.show()

if __name__ =='__main__':
    #calculate dielectric constant at various frequencies
    freq_range = np.array([750e12,250e12])
    ev_max,ev_min = freq_to_ev(freq_range)
    #***Indium Phosphide***
    file_name = 'inp.csv'
    inp.inp(ev_min,ev_max,NUM_POINTS,file_name)
    #***Gold- Lorentz-Drude Model***
    # file_name = 'outLD.csv'
    # au_LD_model.au_model(ev_min,ev_max,NUM_POINTS,file_name)
    #Gold - Brendel-Bormann model
    # file_name = 'outBB.csv'
    # au_BB_model.BB_model(ev_min,ev_max,NUM_POINTS,file_name)
    data = np.genfromtxt(file_name, delimiter=',')
    #main exectution
    # print(recipro(a1,a2))
    eps(data,freq_range)
