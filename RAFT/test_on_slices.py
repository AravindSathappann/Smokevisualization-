import test_raft
import numpy as np
import torch
from torchvision.utils import flow_to_image

test_dests = []
raw_image_dests = []
ground_truth_dests = []
for i in range(102, 117):
    raw_image_dests.append(f"../3D_to_2D_conversion/density_images/density_000{i}_Z50.png")
    ground_truth_dests.append(f"../3D_to_2D_conversion/velocity_slices/velocity_000{i}_Z50.npz")
    
for i in range(len(raw_image_dests)-1):
    test_dests.append((raw_image_dests[i], raw_image_dests[i+1], ground_truth_dests[i]))
    
    #print(test_dests)

data = np.load(ground_truth_dests[0])

print(data.files)
ux = data["vx"]
uy = data["vy"]


#data is 178 pixels  = 0.9 m = height, 100 = width = 0.5 m

#m/s -> pix/s ->
                #width -> 0.5 m -> 200 pixels per meter
                #height -> 0.9 m -> 200 pixels per meter -> non uniform because not square


stacked = np.stack((uy, ux), axis=2)
stacked_in_pixels  = stacked * 200
print(stacked.shape)


# flow shape: (H, W, 2)
flow_tensor = torch.from_numpy(stacked).float()

# Convert to (1, 2, H, W)
flow_tensor = flow_tensor.permute(2, 0, 1).unsqueeze(0)

# Convert to RGB
flow_img = flow_to_image(flow_tensor)[0]




    
    
