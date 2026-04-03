from pathlib import Path
import numpy as np

mask_folder_1 = Path(r"C:\Users\munee\Desktop\Kaggle\RECODAI\recodai-luc-scientific-image-forgery-detection\train_masks")
mask_folder_2 = Path(r"C:\Users\munee\Desktop\Kaggle\RECODAI\recodai-luc-scientific-image-forgery-detection\supplemental_masks")

mask_files = sorted(list(mask_folder_1.glob("*.npy")) + list(mask_folder_2.glob("*.npy")))

layer_counts = []

for mask_path in mask_files:
    mask = np.load(mask_path)

    if mask.ndim == 2:
        layer_counts.append(1)
    elif mask.ndim == 3:
        layer_counts.append(mask.shape[0])
    else:
        raise ValueError(f"Unexpected mask shape {mask.shape} in {mask_path}")

print("Total mask files:", len(mask_files))
print(f"Layer range found in masks: {min(layer_counts)} to {max(layer_counts)}")