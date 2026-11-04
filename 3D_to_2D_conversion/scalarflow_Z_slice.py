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


# TODO: DOWNLOAD sim_000000 FROM THIS LINK TO RUN THIS LOCALLY:
#   https://dataserv.ub.tum.de/s/m1521788
# this path will be different depending on local device
folder = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\3d_to_2d\sim_000000_extracted\sim_000000"

# print("SCALARFLOW FILES: \n")
# for root, dirs, files in os.walk(folder):
#    for file in files:
#        print(os.path.join(root, file))
        
        
# print("VELOCITY FILES:\n")
# for root, dirs, files in os.walk(folder):
#     for file in files:
#         if "velocity" in file.lower():
#             print(os.path.join(root, file))
            

# STORE velocity_000102.npz path
velocity_path = None
for root, dirs, files in os.walk(folder):
    if "velocity_000102.npz" in files:
        velocity_path = os.path.join(root, "velocity_000102.npz")
        break

# load velocity file into numpy
data = np.load(velocity_path)

#print("Contents:", data.files)

for key in data.files:
    array = data[key]
    print("Key:", key)
    print("Shape (size along each dimension):", array.shape)
    print("Data type:", array.dtype)


# Load the velocity field
velocity = data['data']

z_index = 50

# Extract the 2D slice
z_slice = velocity[:, :, z_index, :]

vx = z_slice[:, :, 0] #all x-velocities
vy = z_slice[:, :, 1] #all y-velocities
vz = z_slice[:, :, 2] #all z-velocities

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