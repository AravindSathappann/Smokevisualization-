from scalarflow_utils import (load_file, get_velocity_z_slice, get_density_z_slice)
import numpy as np
import matplotlib.pyplot as plt
import os

###########
##
## For anyone who wants to run this locally to generate their own smoke/velocity images/fields:
##   Change the following output paths to match your own device's file system
##   Change the sim_folder variable to match a locally downloaded sim folder from the dataset
##  - Jess 11/17/26
##
###########

# path to locally downloaded sim folder
sim_folder = folder = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\sim_000000_data\sim_000000_extracted\sim_000000"


# Output folders (My, jess's, local path, for generating the images before uploading to git)
density_output_path = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\Smokevisualization_Github\3D_to_2D_conversion\density_images"
os.makedirs(density_output_path, exist_ok=True)

velocity_output_path = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\Smokevisualization_Github\3D_to_2D_conversion\velocity_slices"
os.makedirs(velocity_output_path, exist_ok=True)

velocity_image_path = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\Smokevisualization_Github\3D_to_2D_conversion\visualized_velocity_images"
os.makedirs(velocity_image_path, exist_ok=True)



## generates a smoke image from a density_file in sim_000000 at a given Z-index
## density_file     : string - ex, "density_000102.npz" 
## z_index          : number - ex, 50
## sim_folder       : local path to sim folder
def generate_smoke_image(density_file, z_index, sim_folder):
    # Load the density field
    density = load_file(density_file, sim_folder)
    #print(density_file, "Shape: ", density.shape)
    
    # Extract the 2D density slice
    z_slice = get_density_z_slice(density, z_index)
    #print (density_file, "slice at Z=", z_index, "Shape: ", z_slice.shape)

    ## make image name 
    ## ex - density_000102_Z50.png for Z = 50
    image_name = density_file.replace(".npz", f"_Z{z_index}.png")
    image_path = os.path.join(density_output_path, image_name)
    
    ## save image to folder
    plt.imsave(
        image_path,
        z_slice.T,
        cmap="gray"
    )


## generates the 2D x * y velocity field from a velocity_file in sim_000000 at a given Z-index
## velocity_file    : string - ex, "velocity_000102.npz" 
## z_index          : number - ex, 50
## sim_folder       : local path to sim folder
##
## which can then have its x and y components accessed via:
##      data = np.load("velocity_000102_Z50.npz")
##      vx = data["vx"]
##      vy = data["vy"]
def generate_2D_velocity_slice(velocity_file, z_index, sim_folder):
    # Load the velocity field
    velocity = load_file(velocity_file, sim_folder)

    # Extract the 2D velocity slide
    z_slice = get_velocity_z_slice(velocity, z_index)

    # all x and y velocities
    vx = z_slice[:, :, 0]
    vy = z_slice[:, :, 1]

    ## make image name
    ## ex - velocity_000102_Z50 for Z= 50
    output_name = velocity_file.replace(".npz", f"_Z{z_index}.npz")
    output_file = os.path.join(velocity_output_path, output_name)

    # save 2D velocity field to folder
    np.savez(output_file, vx=vx.T, vy=vy.T)


## generates visualized velocity field for a velocity file at a specified z-value
## velocity_file    : string - ex, "velocity_000102.npz" 
## z_index          : number - ex, 50
## max_vel          : number - to standardize plotting
## sim_folder       : local path to sim folder
def generate_velocity_image(velocity_file, z_index, max_vel, sim_folder):
    # Load the velocity field
    velocity = load_file(velocity_file, sim_folder)
    
    # Extract the 2D velocity slice
    z_slice = get_velocity_z_slice(velocity, z_index)
    
    vx = z_slice[:, :, 0] #all x-velocities
    vy = z_slice[:, :, 1] #all y-velocities
    #vz = z_slice[:, :, 2] #all z-velocities
    # print("velocity shape", velocity.shape)
    # print("2D slice shape:", z_slice.shape)
    # print("Vx shape:", vx.shape)
    # print("Vy shape:", vy.shape)
    # print("Vz shape:", vz.shape)
    
    # Calculate the 2D velocity magnitude
    magnitude = np.sqrt(vx**2 + vy**2)
   
    # Create the plot
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Plot velocity magnitude as the background
    image = ax.imshow(
        magnitude.T,
        origin="upper",
        cmap="inferno",
        vmin=0,
        vmax=max_vel
    )
    
    # makes grid for velocity arrows
    X, Y = np.meshgrid(
        np.arange(vx.shape[0]),
        np.arange(vx.shape[1]),
    )
    
    # plot arrows every 5 grid points
    ax.quiver(
        X[::5, ::5],
        Y[::5, ::5],
        vx.T[::5, ::5],
        vy.T[::5, ::5],
        color="lime",
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.003,
        headwidth=4
    )
    
    # Add labels and colorbar
    fig.colorbar(image, ax=ax, label="2D Velocity Magnitude")
    ax.set_title(f"{velocity_file} at Z = {z_index}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # make image name
    image_name = velocity_file.replace(".npz", f"_Z{z_index}.png")
    image_path = os.path.join(velocity_image_path, image_name)
    
    # save and close figure
    fig.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)



Z_Index = 50
first_image_index = 102 # INCLUSIVE
last_image_index = 117  # EXCLUSIVE

# this was for velocity field visualization, not needed for just 2D velocity slices.
global_max_vel = 0
for x in range(first_image_index, last_image_index):
    velocity_file_name = f"velocity_000{x}.npz"
    velocity = load_file(velocity_file_name, sim_folder)
    z_slice = get_velocity_z_slice(velocity, Z_Index)

    vx = z_slice[:, :, 0]
    vy = z_slice[:, :, 1]

    magnitude = np.sqrt(vx**2 + vy**2)
    global_max_vel = max(global_max_vel, np.max(magnitude))


#############################################################################
## LOOP TO GENERATE SMOKE IMAGES, VELOCITY FIELD DATA, AND VELOCITY IMAGES ##
#############################################################################
for x in range(first_image_index, last_image_index):
    density_file_name = f"density_000{x}.npz"
    generate_smoke_image(density_file_name, Z_Index, sim_folder)
    
    velocity_file_name = f"velocity_000{x}.npz"
    generate_2D_velocity_slice(velocity_file_name, Z_Index, sim_folder)
    
    # this was just for the velocity field animation for 11/11's meeting
    #generate_velocity_image(velocity_file_name, Z_Index, global_max_vel)







# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# debugging to figure out if dimensions of density/velocity slices match ~
"""
# used this to generate the images to visually check that they match up, they did
#generate_smoke_image("density_000102.npz", Z_Index)
#generate_2D_velocity_slice("velocity_000102.npz", Z_Index)
#generate_velocity_image("velocity_000102.npz", Z_Index, global_max_vel)


# made sure slices from generate_slice fuctions align, they do
density = load_file("density_000102.npz")
velocity = load_file("velocity_000102.npz")
density_slice = get_density_z_slice(density, Z_Index)
velocity_slice = get_velocity_z_slice(velocity, Z_Index)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Density slice:", density_slice.shape)
print("Velocity slice:", velocity_slice.shape)
print("Velocity X:", velocity_slice[:, :, 0].shape)
print("Velocity Y:", velocity_slice[:, :, 1].shape)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# output:
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density slice: (100, 178)
# Velocity slice: (100, 178, 3)
# Velocity X: (100, 178)
# Velocity Y: (100, 178)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# checked if density image and velocity field dimensions match - they didn't
# load saved density image
density_image_path = os.path.join(density_output_path, "density_000102_Z50.png")
density_image = plt.imread(density_image_path)
from PIL import Image
img = Image.open(density_image_path)
# load saved velocity field
velocity_file_path = os.path.join(velocity_output_path, "velocity_000102_Z50.npz")
velocity_data = np.load(velocity_file_path)
vx = velocity_data["vx"]
vy = velocity_data["vy"]
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("density image shape:", density_image.shape)
print("image array shape:", np.array(img).shape)
print("image size (width, height):", img.size)
print("velocity X shape:", vx.shape)
print("velocity Y shape:", vy.shape)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# density image shape: (178, 100, 4)
# image array shape: (178, 100, 4)
# image size (width, height): (100, 178)
# velocity X shape: (100, 178)
# velocity Y shape: (100, 178)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
# here i transposed the velocity x and y shapes to match the transposed density field when i generate the 2D velocity slices, and that made them match up



# extracts all z-slices (as in, generates smoke images & 2D celocity fields) for all density & velocity files in a given path to a downloaded, extracted sim_folder
# sim_folder        : local path to sim folder
def extract_simulation_slices(sim_folder, density_output_path, velocity_output_path):
    os.makedirs(density_output_path, exist_ok=True)
    os.makedirs(velocity_output_path, exist_ok=True)
    
    # loop through files in sim_folder
    for root, dirs, files in os.walk(sim_folder):
        for file in files:
            
            # skip non .npz files
            if not file.endswith(".npz"):
                continue
            
            # load data (to get z-dimension)
            data = load_file(file, sim_folder)
            
            if file.startswith("density_"): # generate smoke images
                for z in range(data.shape[2]):
                    generate_smoke_image(file, z, sim_folder)
            elif file.startswith("velocity_"): # extract velocity slices
                for z in range(data.shape[2]):
                    generate_2D_velocity_slice(file, z, sim_folder)
            else:
                continue
            
            

                    
            
            












