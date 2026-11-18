import test_raft
import numpy as np
import torch
from torchvision.utils import flow_to_image
import matplotlib.pyplot as plt

test_dests = []
raw_image_dests = []
ground_truth_dests = []
for i in range(102, 117):
    raw_image_dests.append(f"../3D_to_2D_conversion/density_images/density_000{i}_Z50.png")
    ground_truth_dests.append(f"../3D_to_2D_conversion/velocity_slices/velocity_000{i}_Z50.npz")
    
for i in range(1, len(raw_image_dests)):
    test_dests.append((raw_image_dests[i-1], raw_image_dests[i], ground_truth_dests[i]))
    
    #print(test_dests)




#data is 178 pixels  = 0.9 m = height, 100 = width = 0.5 m

#m/s -> pix/s ->
                #width -> 0.5 m -> 200 pixels per meter
                #height -> 0.9 m -> 200 pixels per meter -> non uniform because not square


for i in range(len(test_dests)):
    pred = test_raft.calculate_from_images(test_dests[i][0], test_dests[i][1], str(i))
    
    data = np.load(test_dests[i][2])
    ux = data["vx"] 
    uy = data["vy"] 
    stacked = np.stack((ux, uy), axis=2)
    #stacked_in_pixels  = stacked
    print(stacked.shape)

    # flow shape: (H, W, 2)
    flow_tensor = torch.from_numpy(stacked).float()

    # Convert to (1, 2, H, W)
    gt = flow_tensor.permute(2, 0, 1).unsqueeze(0)
    
    gt_vis = gt[0].detach().clone().float().cpu()

    # Replace NaN and Inf values for visualization only
    gt_vis = torch.nan_to_num(
        gt_vis,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )
    flow_rgb = flow_to_image(gt_vis)

    # plt.figure(figsize=(10, 6))

    # plt.imshow(flow_rgb.permute(1, 2, 0).numpy())

    # plt.title("Ground Truth Flow")
    # plt.axis("off")

    # plt.savefig(f"{i}_truth.png")
    
    # Extract flow components
    print("shapes!")
    print(pred.shape)
    print(gt_vis.shape)
    print("predicted states in order pred.abs().max(), pred.abs().mean(), pred.abs().flatten().quantile(0.99): ")
    print(pred.abs().max(), pred.abs().mean(), pred.abs().flatten().quantile(0.99))
    U1 = gt_vis[0, :, :].numpy()
    V1 = gt_vis[1, :, :].numpy()

    U2 = pred[0, :, :].numpy()
    V2 = pred[1, :, :].numpy()
    
    mag1 = np.sqrt(U1**2 + V1**2)
    mag2 = np.sqrt(U2**2 + V2**2)

    H, W = U1.shape

    # Coordinate grids (shared)
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    step = 5

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    image = plt.imread(test_dests[i][0])
    axes[0].imshow(image, cmap='gray', origin='upper')
    axes[1].imshow(image, cmap='gray', origin='upper')

    # First subplot
    q1 = axes[0].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U1[::step, ::step],
        V1[::step, ::step],
        mag1[::step, ::step],
        angles='xy',
        scale_units='xy',
        scale=0.5,
        cmap='viridis'
    )

    # Second subplot
    q2 = axes[1].quiver(
        X[::step, ::step],
        Y[::step, ::step],
        U2[::step, ::step],
        V2[::step, ::step],
        mag2[::step, ::step],
        angles='xy',
        scale_units='xy',
        scale=0.5,
        cmap='viridis'
    )

    # Apply identical axis settings
    for ax in axes:
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xlim(0, W)
        ax.set_ylim(H, 0)

    axes[0].set_title("Ground Truth")
    axes[1].set_title("RAFT Prediction")
    
    vmax = max(mag1.max(), mag2.max())

    q1.set_clim(0, vmax)
    q2.set_clim(0, vmax)

    cbar_ax = fig.add_axes([0.02, 0.15, 0.015, 0.7])

    fig.colorbar(q1, cax=cbar_ax, label='Velocity Magnitude (pixels/frame)')

    plt.tight_layout()
    plt.savefig(f"comparison{i}.png", dpi=300)
    
    
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
    
    




    
    
