import matplotlib.pyplot as plt
import torch


def visualize_sample(dataset, idx):
    """
    Visualize one sample from the dataset:
    1. original image
    2. binary mask
    3. overlay of mask on image

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Your dataset object.
    idx : int
        Index of the sample to visualize.
    """

    sample = dataset[idx]

    image = sample[0]
    mask = sample[1]

    # Convert image from (C, H, W) to (H, W, C) for matplotlib
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).cpu().numpy()

    # Convert mask from (1, H, W) to (H, W)
    if isinstance(mask, torch.Tensor):
        mask = mask.squeeze(0).cpu().numpy()

    # Make sure mask is binary
    mask = (mask > 0).astype(float)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Image
    axes[0].imshow(image)
    axes[0].set_title("Processed Image")
    axes[0].axis("off")

    # 2. Mask
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Processed Mask")
    axes[1].axis("off")

    # 3. Overlay
    axes[2].imshow(image)
    axes[2].imshow(mask, cmap="Reds", alpha=0.4)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()