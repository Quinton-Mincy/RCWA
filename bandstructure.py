#Author(s): Quinton Mincy
#Last Modified: 01/10/24
#Code Built Arround Alex Song's Examples for his Inkstone library: https://github.com/alexysong/inkstone



import numpy as np
from numpy import linalg as la

from matplotlib import pyplot as plt
from matplotlib import cm
import au_LD_model
import math
from inkstone import Inkstone
from tqdm import tqdm
import csv



#program parameters
c    = 299792458
#should be multiples of 90
num_points = 90
# latc = 0.3
# a1 = (-(1/2)*latc,latc*math.sqrt(3)/2)
# a2 = ((1/2)*latc,latc*math.sqrt(3)/2 )
latc = 1
a1=(1,0)
a2=(0,1)


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
    #kx,ky
    #k = np.array( [(ω/c) * math.sin(theta)*math.cos(phi) , (ω/c) * math.sin(theta)*math.sin(phi) ] )
    
    kx =  ω * np.cos(np.pi/2 - theta) * np.cos(phi)
    ky =  ω * np.cos(np.pi/2 - theta) * np.sin(phi)
    k = np.array([kx,ky])
  
    return k

# def recpV(a1,a2,k):
#     Q = np.array([[0,-1],[1,0]])
#     b1 = (2*math.pi) * ( np.matmul(Q,a2) / np.dot(a1,np.matmul(Q,a2)) )
#     b2 = (2*math.pi) * ( np.matmul(Q,a1) / np.dot(a2,np.matmul(Q,a1)) )

#     k1 = np.dot(k,b1) / np.dot(b1,b1)
#     k2 = np.dot(k,b2) / np.dot(b2,b2)

#     # k_recp = np.array([k1*b1, k2*b2 ])
#     k_recp = np.array([b1,b2])
    
#     return k_recp


def plot_spectrum(epsilon,frequency):
    s = Inkstone()
    s.lattice = (a1,a2)
    s.num_g = 50

    # s.AddMaterial(name='Au', epsilon=epsilon[0])
    s.AddMaterial(name='Au', epsilon=12)
    s.AddLayer(name='in', thickness=0, material_background='vacuum')
    s.AddLayer(name='slab', thickness=0.5*latc, material_background='Au')
    s.AddLayerCopy(name='out', original_layer='in', thickness=0)
    s.AddPatternDisk(layer='slab', pattern_name='disk', material='vacuum', radius=0.2*latc)

    #set figure size
    plt.figure(figsize=(10, 6))

    #sweep through incident angle
    theta_values = np.linspace(0,90,num_points)

    transmission_values = []
    kx_values = []

    for theta in tqdm(theta_values):

        s.SetExcitation(theta=theta, phi=0, s_amplitude=1/np.sqrt(2), p_amplitude=1/np.sqrt(2))
        flux_in = []
        flux_out = []

        for i, nu in enumerate(frequency):
            # Update material properties and frequency
            # s.SetMaterial(name='Au', epsi=epsilon[i])
            s.SetMaterial(name='Au', epsi=12)
            s.SetFrequency(nu)
            
            flux_in.append(s.GetPowerFlux('in'))
            flux_out.append(s.GetPowerFlux('out'))


        incident = np.array([a[0] for a in flux_in])
        reflection = -np.array([a[1] for a in flux_in]) / incident
        transmission = np.array([a[0] for a in flux_out]) / incident

        wavelengths = freq2lambda(1, frequency) * 1000  # Convert to nm

        transmission_values.append(transmission)

        # Plot transmission vs wavelength
        plt.plot(wavelengths, transmission * 100, label=f'Theta = {theta}°')

    #at each frequency, calculate projection of incident wavevectors onto reciprocal lattice
    for nu in frequency:
        wavevectors = [wavevector(ang,0,nu) for ang in theta_values]

        k_recps =  recipro(a1,a2)

        projections = [proj(wv,k_recps[0]) for wv in wavevectors] 
        proj_mags = [np.linalg.norm(proj) for proj in  projections]
        kx_values.append(proj_mags)
    
    
    #Transmision vs wavelength
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Transmission (%)')
    plt.title('Transmission vs Wavelength for Various Theta Values')
    plt.legend(loc='best')  # Automatically place legend
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to prevent clipping

    # Prepare data for contour plot
    kx_values = np.array(kx_values)
    transmission_values = np.array(transmission_values)
    transmission_values_2d = transmission_values.T 
    frequency_2d = np.tile(frequency[:, np.newaxis], (1, kx_values.shape[1])) 

    #image show transmission
    plt.figure(figsize=(10, 6))
    plt.imshow(transmission_values_2d,cmap = 'viridis')
    plt.tight_layout()


    #Contour map
    plt.figure(figsize=(10, 6))
    plt.contourf(kx_values, frequency_2d, transmission_values_2d, levels=100, cmap='viridis')  # Colormap
    plt.colorbar(label='Transmission')
    plt.xlabel('$k_x$')
    plt.ylabel('Frequency ')
    plt.title('Transmission Intensity Contour ($k_x$ vs. Wavelength)')
    plt.tight_layout()


def extract(data):
    points = np.array((data[:,0],data[:,1],data[:,2],data[:,3],data[:,4]))
    return points


def eps(data):
    points = extract(data)
    #wavelength
    lam = points[0]
    #index of frefraction
    n   = points[1]
    #extinction coeeficient
    k   = points[2]
    #dielectric constant
    ε_real   = points[3]
    ε_imag   = points[4]
       
    frequency = np.linspace(.25,.6,num_points)
    epsilon = [complex(real, imag) for real, imag in zip(ε_real, ε_imag)]  
    plot_spectrum(epsilon,frequency)
 

    plt.show()

if __name__ =='__main__':
    au_LD_model.au_model()
    data = np.genfromtxt('out.csv', delimiter=',')
    eps(data)