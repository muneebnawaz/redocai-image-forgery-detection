from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---- choose one forged example manually ----
# Replace these with one real forged image and its matching mask
image_path = Path("recodai-luc-scientific-image-forgery-detection/train_images/forged/10.png")
mask_path = Path("recodai-luc-scientific-image-forgery-detection/train_masks/10.npy")

# ---- load image ----
image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---- load mask ----
mask = np.load(mask_path)

# Collapse stacked masks if needed
if mask.ndim == 3:
    mask = np.any(mask > 0, axis=0)
elif mask.ndim == 2:
    mask = mask > 0
else:
    raise ValueError(f"Unexpected mask shape: {mask.shape}")

mask = mask.astype(np.float32)

print("Original image shape:", image.shape)
print("Original mask shape:", mask.shape)
print("Original mask unique values:", np.unique(mask))

# ---- resize target ----
target_size = (256, 256)   # (width, height) for cv2.resize

# Resize image normally
image_resized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

# Resize mask two different ways
mask_nearest = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
mask_bilinear = cv2.resize(mask, target_size, interpolation=cv2.INTER_LINEAR)

print("\nNearest-neighbor resized mask unique values:")
print(np.unique(mask_nearest))

print("\nBilinear resized mask unique values:")
print(np.unique(mask_bilinear)[:20])  # show first few only
print("Total unique bilinear values:", len(np.unique(mask_bilinear)))

# Optional: re-binarize nearest result
mask_nearest_binary = (mask_nearest > 0).astype(np.float32)

# ---- plot everything ----
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].imshow(image)
axes[0, 0].set_title("Original Image")
axes[0, 0].axis("off")

axes[0, 1].imshow(mask, cmap="gray")
axes[0, 1].set_title("Original Mask")
axes[0, 1].axis("off")

axes[0, 2].imshow(image_resized)
axes[0, 2].set_title("Resized Image (bilinear)")
axes[0, 2].axis("off")

axes[1, 0].imshow(mask_nearest, cmap="gray")
axes[1, 0].set_title("Mask Resized (nearest)")
axes[1, 0].axis("off")

axes[1, 1].imshow(mask_bilinear, cmap="gray")
axes[1, 1].set_title("Mask Resized (bilinear)")
axes[1, 1].axis("off")

axes[1, 2].imshow(mask_nearest_binary, cmap="gray")
axes[1, 2].set_title("Nearest Mask Re-binarized")
axes[1, 2].axis("off")

plt.tight_layout()
plt.show()