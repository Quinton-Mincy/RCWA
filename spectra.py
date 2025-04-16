#Author(s): Quinton Mincy
#Last Modified: 04/16/25
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
#should be multiples of 3
NUM_POINTS = 180
NUM_G = 50
#microns
thickness = 1

#program parameters
# latc = 0.3
# latc_nm = 300e-9
latc = 0.5
latc_nm = 500e-9
a1 = (latc,0)
a2 = ((1/2)*latc,latc*math.sqrt(3)/2)


def init_inkstone(epsilon,core,shell,thickness):
    s = Inkstone()
    s.lattice = ((a1,a2))
    s.num_g = NUM_G

    s.AddMaterial(name='InP', epsilon=epsilon)
    s.AddMaterial(name='Al2O3', epsilon=2.9467)

    # s.AddLayer(name='in', thickness=0, material_background='vacuum')

    # s.AddLayer(name='wire', thickness=thickness, material_background='vacuum') 
    # # Add the outer Al2O3 shell
    # s.AddPatternPolygon(layer='wire', material='Al2O3', pattern_name='shell', vertices=shell)
    # # Add the inner InP core (must be completely inside the shell)
    # s.AddPatternPolygon(layer='wire', material='InP', pattern_name='core', vertices=core)
    # s.AddLayer(name='out', thickness=0, material_background='InP')

    # s.AddLayer(name='in', thickness=0, material_background='vacuum')
    # s.AddLayer(name='shell', thickness=0.01, material_background='Al2O3')  # outer shell is background
    # s.AddLayer(name='wire', thickness=thickness, material_background='InP')  # outer shell is background
    # s.AddLayer(name='out', thickness=0, material_background='InP')  # substrate
    # s.AddPatternPolygon(layer="wire", material="InP", pattern_name="wire", vertices=core)

    
    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='shell', thickness=0.01, material_background='vacuum')
    s.AddLayer(name='wire', thickness=thickness, material_background='Al2O3')
    s.AddLayer(name='out', thickness=0, material_background='InP')
    s.AddPatternPolygon(layer="shell", material="Al2O3", pattern_name="poly1",
        vertices=shell)
    s.AddPatternPolygon(layer="wire", material="InP", pattern_name="poly2",
    vertices=core)

    # s.AddLayer(name='in', thickness=0, material_background='InP')
    # s.AddLayer(name='wire', thickness=thickness, material_background='InP')
    # s.AddLayer(name='out', thickness=0, material_background='vacuum')
    # s.AddPatternPolygon(layer="wire", material="InP", pattern_name="poly1",
    #     vertices=core)

    # s.AddLayer(name='in', thickness=0, material_background='vacuum')
    # s.AddLayer(name='wire', thickness=thickness, material_background='InP')
    # s.AddLayer(name='out', thickness=0, material_background='vacuum')
    # s.AddPatternPolygon(layer="wire", material="InP", pattern_name="poly1",
    #     vertices=core)
    return s

def plot_spectrum(ax, wavelength, transmission,reflection,absorbance, thickness):
    # Plot transmission vs. wavelength on the provided axes.
    # Using a label to differentiate different thicknesses.
    ax.plot(wavelength, transmission * 100, label='transmission')
    ax.plot(wavelength, reflection * 100, label='reflection')
    ax.plot(wavelength, absorbance * 100, label='absorbance')


def RCWA(epsilon,frequency,wavelength):
    #normalize frequency for RCWA calculations
    frequency_norm = normalize_frequency(frequency,(latc_nm))
    #init geometry
    shell = generate_hexagon(center=(0, 0), radius=118e-3)  # microns
    core = generate_hexagon(center=(0, 0), radius=108e-3)
    thickness = [1]

    fig, ax = plt.subplots(figsize=(10, 6))

    for thick in tqdm(thickness):
        #arrays for incidence, transmission, refelction
        flux_in = []
        flux_out = []

        for i,nu in enumerate(frequency_norm):
            s = init_inkstone(epsilon[i], core, shell, thick) 
            # s = init_inkstone(11.56, core, shell, thick) 

            s.SetExcitation(theta=6, phi=0, s_amplitude=1, p_amplitude=0)
            s.SetFrequency(nu)
            
            flux_in.append(s.GetPowerFlux('in'))
            flux_out.append(s.GetPowerFlux('out'))


        incident = np.array([a[0] for a in flux_in])
        reflection = -np.array([a[1] for a in flux_in]) / incident
        transmission = np.array([a[0] for a in flux_out]) / incident
        absorbance = 1 - transmission - reflection


        plot_spectrum(ax,wavelength,transmission,reflection,absorbance,thick)
    
    # After the loop, label and display the figure
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Transmission (%)')
    ax.set_title('Transmission vs Wavelength for Various Thicknesses')
    ax.grid(True)
    # ax.set_xlim(500, 1000)

    # ax.set_ylim(0, 100)
    ax.legend(title='Thickness')
    plt.tight_layout()
    plt.show()


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


# Function to generate hexagon vertices
def generate_hexagon(center,radius):

    cx, cy = center
    vertices = [
        (cx + radius * np.cos(2 * np.pi * i / 6), cy + radius * np.sin(2 * np.pi * i / 6)) for i in range(6)]
    return vertices


def eps(data,freq_range):
    points = extract(data)
    #dielectric constant
    ε_real   = points[3]
    ε_imag   = points[4]
    epsilon = [complex(real, imag) for real, imag in zip(ε_real, ε_imag)]  
    epsilon = epsilon
    #initialize frequency array
    freq_params = np.linspace(freq_range[0],freq_range[1], NUM_POINTS)
    wavelength = (c_nm / freq_params)  # λ = c / f (in nm)
    wavelength = wavelength
    #RCWA
    RCWA(epsilon,freq_params,wavelength)

    plt.show()

if __name__ =='__main__':

    #LD model will calculate dielectric constant at various frequencies
    freq_range = np.array([1000e12,250e12])
    ev_max,ev_min = freq_to_ev(freq_range)
    #calculate epsilon

    #***Indium Phosphide***
    file_name = 'inp.csv'
    inp.inp(ev_min,ev_max,NUM_POINTS,file_name)
    #***Gold***
    # file_name = 'outBB.csv'
    # au_BB_model.BB_model(ev_min,ev_max,NUM_POINTS,file_name)

    data = np.genfromtxt(file_name, delimiter=',')
    #main exectution
    eps(data,freq_range)
