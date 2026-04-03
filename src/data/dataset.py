from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BiomedicalForgeryDataset(Dataset):
    def __init__(self, df, transforms=None):
        """
        PyTorch Dataset for biomedical copy-move forgery segmentation.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame created by load_dataset.py
        transforms : callable, optional
            Albumentations-style transform that takes image and mask
        """
        self.df = df.reset_index(drop=True)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        height, width = image.shape[:2]

        if row["has_mask"]:
            mask_path = Path(row["mask_paths"][0])
            mask = np.load(mask_path)

            if mask.ndim == 2:
                if mask.shape != (height, width):
                    raise ValueError(
                        f"Mask shape {mask.shape} does not match image shape {(height, width)} "
                        f"for case_id={row['case_id']}"
                    )

            elif mask.ndim == 3:
                if mask.shape[1:] != (height, width):
                    raise ValueError(
                        f"Mask spatial shape {mask.shape[1:]} does not match image shape {(height, width)} "
                        f"for case_id={row['case_id']}"
                    )

                mask = np.any(mask > 0, axis=0).astype(np.uint8)

            else:
                raise ValueError(
                    f"Unsupported mask shape {mask.shape} for case_id={row['case_id']}"
                    )
        else:
            mask = np.zeros((height, width), dtype=np.uint8)

        mask = (mask > 0).astype(np.uint8)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.tensor(image, dtype=torch.float32)
            image = image.permute(2, 0, 1) / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        return image, mask