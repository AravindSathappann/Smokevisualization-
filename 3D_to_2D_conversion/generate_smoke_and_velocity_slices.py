from scalarflow_utils import (load_file, get_velocity_z_slice, get_density_z_slice)
import numpy as np
import matplotlib.pyplot as plt
import os

###########
##
## Note for anyone else who wants to run this to generate their own smoke/velocity images/fields:
##   if running this code locally, change the following output paths to match your own device's file system
##   ALSO change the folder variable in scalarflow_utils.py to match a locally downloaded sim folder from the ScalarFlow dataset.
##   This was just ran on my device to generate the 15 smoke images, velocity fields, and velocity images with them
##  - Jess 11/11/26
##
###########

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
def generate_smoke_image(density_file, z_index):
    # Load the density field
    density = load_file(density_file)
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
##
## which can then have its x and y components accessed via:
##      data = np.load("velocity_000102_Z50.npz")
##      vx = data["vx"]
##      vy = data["vy"]
def generate_2D_velocity_slice(velocity_file, z_index):
    # Load the velocity field
    velocity = load_file(velocity_file)

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
    np.savez(output_file, vx=vx, vy=vy)


## generates visualized velocity field for a velocity file at a specified z-value
## velocity_file    : string - ex, "velocity_000102.npz" 
## z_index          : number - ex, 50
## max_vel          : number - to standardize plotting
def generate_velocity_image(velocity_file, z_index, max_vel):
    # Load the velocity field
    velocity = load_file(velocity_file)
    
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



### GENERATES 15 SMOKE IMAGES AND CORRESPONDING 2D VELOCITY FIELDS
Z_Index = 50
first_image_index = 102 # INCLUSIVE
last_image_index = 117  # EXCLUSIVE

# this was for velocity field visualization
global_max_vel = 0
for x in range(first_image_index, last_image_index):
    velocity_file_name = f"velocity_000{x}.npz"
    velocity = load_file(velocity_file_name)
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
    #generate_smoke_image(density_file_name, Z_Index)
    
    velocity_file_name = f"velocity_000{x}.npz"
    #generate_2D_velocity_slice(velocity_file_name, Z_Index)
    
    # this was just for the velocity field animation for 11/11's meeting
    generate_velocity_image(velocity_file_name, Z_Index, global_max_vel)






