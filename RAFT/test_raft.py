import torch
import cv2
import numpy as np
import torch.nn.functional as F

from torchvision.models.optical_flow import (
    raft_large,
    Raft_Large_Weights
)
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt
import h5py

def test_predictions(truth_path, img1_path, img2_path, image_batch):
    # Load Spring ground truth, automatically finding the flow dataset
    with h5py.File(truth_path, "r") as f:
        candidates = []

        def find_flow_dataset(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape = obj.shape

                if len(shape) == 3 and (shape[-1] == 2 or shape[0] == 2):
                    candidates.append(name)

        f.visititems(find_flow_dataset)

        if len(candidates) == 0:
            raise ValueError("No 2D flow dataset with 2 channels found in the .flo5 file.")

        if len(candidates) > 1:
            raise ValueError(
                f"Multiple possible flow datasets found: {candidates}. "
                "Please select the correct one."
            )

        dataset_name = candidates[0]
        print(f"Using dataset: {dataset_name}")

        gt = np.asarray(f[dataset_name])

    # Convert GT to [2, H, W]
    if gt.ndim == 3 and gt.shape[-1] == 2:
        gt = np.moveaxis(gt, -1, 0)
    elif gt.ndim == 3 and gt.shape[0] == 2:
        pass
    else:
        raise ValueError(f"Unexpected GT shape: {gt.shape}")

    gt = torch.from_numpy(gt.astype(np.float32))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    weights = Raft_Large_Weights.DEFAULT

    model = raft_large(weights=weights)

    model = model.to(device)

    model.eval()

    print("reading images!")
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    print("converting images!")
    # OpenCV loads images as BGR.
    # RAFT expects RGB.
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # Convert NumPy arrays to PyTorch tensors
    img1 = torch.from_numpy(img1).permute(2, 0, 1)
    img2 = torch.from_numpy(img2).permute(2, 0, 1)

    # Add batch dimension
    img1 = img1.unsqueeze(0)
    img2 = img2.unsqueeze(0)

    if (img1.size()[2] % 8 != 0 or 
        img1.size()[3] % 8 != 0 or
        img2.size()[2] % 8 != 0 or 
        img2.size()[3] % 8 != 0):
        print("wrong image dim, must be divisible by 8!")
        quit()
    #ensure dimensions are divisible by 8
    print("transforming weights!")
    transforms = weights.transforms()

    img1, img2 = transforms(img1, img2)

    print("predicting!")
    with torch.no_grad():

        flow_predictions = model(
            img1.to(device),
            img2.to(device)
        )
        
        flow = flow_predictions[-1]
        pred = flow[0].detach().cpu()
        
        flow_rgb = flow_to_image(flow)[0]

        plt.figure(figsize=(10, 6))

        plt.imshow(flow_rgb.permute(1, 2, 0).cpu().numpy())

        plt.title("RAFT Optical Flow")
        plt.axis("off")

        plt.savefig("predictions/predicted_" + str(image_batch)+ ".png")
        
        # Resize GT's spatial grid to RAFT resolution `if necessary.
        # Important: do NOT scale u/v magnitudes here.
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(
                gt.unsqueeze(0),
                size=pred.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )[0]
        print(gt.size())
        
        # Ensure GT is float32
        gt_vis = gt.detach().clone().float().cpu()

        # Replace NaN and Inf values for visualization only
        gt_vis = torch.nan_to_num(
            gt_vis,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )
        flow_rgb = flow_to_image(gt_vis)

        plt.figure(figsize=(10, 6))

        plt.imshow(flow_rgb.permute(1, 2, 0).numpy())

        plt.title("RAFT Optical Flow")
        plt.axis("off")

        plt.savefig("predictions/ground_truth_flow_" + str(image_batch) + ".png")
        

        # Valid pixels: at minimum, remove NaN/Inf values.
        valid = torch.isfinite(gt).all(dim=0) & torch.isfinite(pred).all(dim=0)

        # Endpoint error, in pixels
        epe = torch.linalg.vector_norm(pred - gt, dim=0)

        epe_valid = epe[valid]
        
        gt_mag = torch.linalg.vector_norm(gt, dim=0)
        pred_mag = torch.linalg.vector_norm(pred, dim=0)

        print("GT flow magnitude:")
        print("Mean:", gt_mag.mean().item())
        print("Median:", gt_mag.median().item())
        print("Max:", gt_mag.max().item())
        
        print("GT contains NaN:", torch.isnan(gt).any().item())
        print("GT contains Inf:", torch.isinf(gt).any().item())

        print("NaN count:", torch.isnan(gt).sum().item())
        print("Inf count:", torch.isinf(gt).sum().item())

        print("Total values:", gt.numel())

        print("\nRAFT flow magnitude:")
        print("Mean:", pred_mag.mean().item())
        print("Median:", pred_mag.median().item())
        print("Max:", pred_mag.max().item())

        print(f"Valid pixels: {valid.sum().item()}")
        print(f"Mean EPE:     {epe_valid.mean().item():.4f} px")
        print(f"Median EPE:   {epe_valid.median().item():.4f} px")
        print(f"90th pct EPE: {torch.quantile(epe_valid, 0.90).item():.4f} px")
        print(f"Max EPE:      {epe_valid.max().item():.4f} px")
    
    
    
# t_path1 = "data/springer/spring/train/0001/flow_FW_left/flow_FW_left_0001.flo5"
# i_path1 = "data/springer/spring/train/0001/frame_left/frame_left_0001.png"
# i_path2 = "data/springer/spring/train/0001/frame_left/frame_left_0002.png"
# test_predictions(t_path1, i_path1, i_path2, 1)

t_path1 = "data/springer/spring/train/0002/flow_FW_left/flow_FW_left_0001.flo5"
i_path1 = "data/springer/spring/train/0002/frame_left/frame_left_0001.png"
i_path2 = "data/springer/spring/train/0002/frame_left/frame_left_0002.png"
test_predictions(t_path1, i_path1, i_path2, 2)

# t_path1 = "data/springer/spring/train/0004/flow_FW_left/flow_FW_left_0001.flo5"
# i_path1 = "data/springer/spring/train/0004/frame_left/frame_left_0001.png"
# i_path2 = "data/springer/spring/train/0004/frame_left/frame_left_0002.png"
# test_predictions(t_path1, i_path1, i_path2, 3)

# t_path1 = "data/springer/spring/train/0005/flow_FW_left/flow_FW_left_0001.flo5"
# i_path1 = "data/springer/spring/train/0005/frame_left/frame_left_0001.png"
# i_path2 = "data/springer/spring/train/0005/frame_left/frame_left_0002.png"
# test_predictions(t_path1, i_path1, i_path2, 4)