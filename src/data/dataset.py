from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BiomedicalForgeryDataset(Dataset):
    def __init__(self, df, target_size=(512, 512), transforms=None):
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
        self.target_size = target_size
        self.transforms = transforms

    def __len__(self):
        return len(self.df)
    
    def _resize_image_and_mask(self, image, mask):
        """
        Resize image and mask to the same target size.

        image: H x W x 3
        mask:  H x W
        """
        target_h, target_w = self.target_size

        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_LINEAR
        )        

        mask = cv2.resize(
            mask,
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST
        )
        return image, mask
    
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

        # Resize image and mask together
        image, mask = self._resize_image_and_mask(image, mask)

        mask = (mask > 0).astype(np.uint8)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(image).float()
            image = image.permute(2, 0, 1) / 255.0

        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        mask = (mask > 0).float()

        return image, mask