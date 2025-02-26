#Author(s): Quinton Mincy
#Last Modified: 01/10/24
#Code Built Arround Alex Song's Examples for his Inkstone library: https://github.com/alexysong/inkstone



import numpy as np
from numpy import linalg as la

from matplotlib import pyplot as plt
from matplotlib import cm
import au_LD_model
import au_BB_model
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
#should be multiples of 3
NUM_POINTS = 90
NUM_G = 100
#microns
thickness = 0.02

#program parameters
latc = 0.3
latc_nm = 300e-9
a1 = (latc,0)
a2 = ((1/2)*latc,latc*math.sqrt(3)/2)


def init_inkstone(epsilon,geometry,thickness):
    s = Inkstone()
    s.lattice = ((a1,a2))
    s.num_g = NUM_G

    s.AddMaterial(name='Au', epsilon=epsilon)
    
    s.AddLayer(name='in', thickness=0, material_background='Au')
    s.AddLayer(name='slab', thickness=thickness, material_background='vacuum')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)

    s.AddPatternPolygon(layer="slab", material="Au", pattern_name="poly1",
        vertices=geometry)

    return s

def plot_spectrum(wavelength, transmission,thickness):
    #set figure size
    plt.figure(figsize=(10, 6))
    # Plot transmission vs wavelength
    plt.plot(wavelength, transmission * 100)

    #Transmision vs wavelength
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.title(f'Transmission vs Wavelength at {thickness*1000} nm')
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping

def RCWA(epsilon,frequency,wavelength):
    #normalize frequency for RCWA calculations
    frequency_norm = normalize_frequency(frequency,(latc_nm))
    #create geometry
    centers=[(0,0),(latc,0),(latc/2,latc*np.sqrt(3)/2 )]
    geometry = init_geometry(latc/2,centers)

    # thickness = [0.015,0.018,0.02,0.025]#microns
    thickness = [0.02]
    for thick in tqdm(thickness):
        #initialize inksone object
        s = init_inkstone(epsilon[0],geometry,thick)
        s.SetExcitation(theta = 0, phi=0, s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
        #arrays for incidence, transmission, refelction
        flux_in = []
        flux_out = []

        for i,nu in enumerate(frequency_norm):
            # Update material properties and frequency
            s.SetMaterial(name='Au', epsi=epsilon[i])
            s.SetFrequency(nu)
            
            flux_in.append(s.GetPowerFlux('in'))
            flux_out.append(s.GetPowerFlux('out'))


        incident = np.array([a[0] for a in flux_in])
        reflection = -np.array([a[1] for a in flux_in]) / incident
        transmission = np.array([a[0] for a in flux_out]) / incident

        plot_spectrum(wavelength,transmission,thick)


def extract(data):
    points = np.array((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]))
    return points

def freq_to_ev(freq):
    h = 6.626e-34
    unit = 1.602e-19
    e = (h*freq)/unit
    return e

def freq2lambda(freq):
    return c/freq

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


def eps(data,freq_range):
    points = extract(data)
    #dielectric constant
    ε_real   = points[3]
    ε_imag   = points[4]
    epsilon = [complex(real, imag) for real, imag in zip(ε_real, ε_imag)]  
    #initialize frequency array
    freq_params = np.linspace(freq_range[0],freq_range[1], NUM_POINTS)
    wavelength = (c_nm / freq_params)  # λ = c / f (in nm)
    #RCWA
    RCWA(epsilon,freq_params,wavelength)

    plt.show()

if __name__ =='__main__':

    #LD model will calculate dielectric constant at various frequencies
    freq_range = np.array([250e12,750e12])
    ev_min,ev_max = freq_to_ev(freq_range)
    #calculate epsilon
    file_name = 'outLD.csv'
    au_LD_model.au_model(ev_min,ev_max,NUM_POINTS,file_name)
    # file_name = 'outBB.csv'
    # au_BB_model.BB_model(ev_min,ev_max,NUM_POINTS,file_name)

    data = np.genfromtxt(file_name, delimiter=',')
    #main exectution
    eps(data,freq_range)
