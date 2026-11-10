# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 12:05:32 2026

@author: sirow
"""

import numpy as np
import matplotlib.pyplot as plt
import os
# print("NumPy:", np.__version__)
# print("Matplotlib:", plt.matplotlib.__version__)
# print("NumPy version:", np.__version__)

folder = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\sim_000000_data\sim_000000_extracted\sim_000000"
### TODO: DOWNLOAD sim_000000 FROM THIS LINK TO RUN THIS LOCALLY: ###
#   https://dataserv.ub.tum.de/s/m1521788
### this path will be different depending on local device         ###

# print("SCALARFLOW FILES: \n")
# for root, dirs, files in os.walk(folder):
#    for file in files:
#        print(os.path.join(root, file))
# print("VELOCITY FILES:\n")
# for root, dirs, files in os.walk(folder):
#     for file in files:
#         if "velocity" in file.lower():
#             print(os.path.join(root, file))
            


## puts velocity or density field data from a .npz file into an x * y * z * 3 array 
##   currently loads file from sim_000000
##   depends on what the folder variable at the top leads to
## file_name        : string - ex, "velocity_000102.npz" 
def load_file(file_name):
    # store path to velocity file
    path = None
    for root, dirs, files in os.walk(folder):
        if file_name in files:
            path = os.path.join(root, file_name)
            break
    if path is None:
        raise FileNotFoundError(f"{file_name} not found")
    
    # load velocity file into numpy
    data = np.load(path)
    
    print("Contents:", data.files)
    for key in data.files:
        array = data[key]
        print("Key:", key)
        print("Shape (size along each dimension):", array.shape)
        print("Data type:", array.dtype)
    
    return data['data']


## returns x * y * 3 array with 3D velocity vectors at each (x, y) point in a Z-slice
## velocity_data    : loaded from load_file
def get_velocity_z_slice(velocity_data, z_index):
    return velocity_data[:, :, z_index, :]


## returns x * y array with density values at each (x, y) point in a Z-slice
## density_data     : loaded from load_file
def get_density_z_slice(density_data, z_index):
    return density_data[:, :, z_index, 0]


## makes a matplotlib plot of velocities in a velocity_file at the given z_index
## velocity_file    : string - ex, "velocity_000102.npz" 
## z_index          : number - ex, 50
def plot_velocity(velocity_file, z_index):
    # Load the velocity field
    velocity = load_file(velocity_file)
    
    # Extract the 2D velocity slice
    z_slice = get_velocity_z_slice(velocity, z_index)
    
    vx = z_slice[:, :, 0] #all x-velocities
    vy = z_slice[:, :, 1] #all y-velocities
    vz = z_slice[:, :, 2] #all z-velocities
    
    print("velocity shape", velocity.shape)
    print("2D slice shape:", z_slice.shape)
    print("Vx shape:", vx.shape)
    print("Vy shape:", vy.shape)
    print("Vz shape:", vz.shape)
    
    # Calculate magnitude of every velocity vector at every point
    magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
    
    # Plot velocity
    plt.figure(figsize=(9, 6)) #matlab plot 9w 6h
    
    # transpose magnitude array so x is horizontal, y is vertical
    # put first row of array at bottom
    # inferno color map
    plt.imshow(magnitude.T, origin='lower', cmap='inferno') 
    
    plt.colorbar(label='Velocity magnitude')
    plt.title(f'Velocity Magnitude at Z = {z_index}')
    plt.xlabel('dimension 1')
    plt.ylabel('dimension 2')
    plt.show()



## makes a matplotlib plot of densities at every point in a density_file at the given z_index
## density_file     : string - ex, "density_000102.npz" 
## z_index          : number - ex, 50
def plot_smoke_image(density_file, z_index):
    # Load the density field
    density = load_file(density_file)
    print(density_file, "Shape: ", density.shape)
    
    # Extract the 2D density slice
    z_slice = get_density_z_slice(density, z_index)
    print (density_file, "slice at Z=", z_index, "Shape: ", z_slice.shape)

    plt.figure(figsize=(9, 6))
    plt.imshow(z_slice.T, origin='lower', cmap='gray')
    
    # plt.colorbar(label='Density')
    # plt.title(f'Density at Z = {z_index}')
    # plt.xlabel('dimension 1')
    # plt.ylabel('dimension 2')
    
    plt.show()



#### plots the velocity from "velocity_000103.npz" at Z-index = 50
#### generates smoke images at density_000102 and 103

#plot_velocity("velocity_000103.npz", 50)
#plot_smoke_image("density_000102.npz", 50)
#plot_smoke_image("density_000103.npz", 50)

    
    
    





