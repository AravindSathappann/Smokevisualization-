from scalarflow_utils import (load_file, get_velocity_z_slice, get_density_z_slice)
import numpy as np
import matplotlib.pyplot as plt
import os


# Output folders (My, jess's, local path, for generating the images before uploading to git)
density_output_path = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\Smokevisualization_Github\3D_to_2D_conversion\density_images"
os.makedirs(density_output_path, exist_ok=True)

velocity_output_path = r"C:\Users\sirow\Documents\WPI Classes & Projects\_MQP\Smokevisualization_Github\3D_to_2D_conversion\velocity_slices"
os.makedirs(velocity_output_path, exist_ok=True)


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



### GENERATES 15 SMOKE IMAGES AND CORRESPONDING 2D VELOCITY FIELDS
Z_Index = 50
for x in range(102, 117):
    density_file_name = f"density_000{x}.npz"
    generate_smoke_image(density_file_name, Z_Index)
    
    velocity_file_name = f"velocity_000{x}.npz"
    generate_2D_velocity_slice(velocity_file_name, Z_Index)






