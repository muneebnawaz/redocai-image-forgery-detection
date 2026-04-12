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

    def _load_image(self, row):
        image_path = Path(row["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def _load_instance_masks(self, row, height, width):
        """
        Load original per-instance masks at original image resolution.

        Returns
        -------
        list of np.ndarray
            Each entry is a binary mask of shape (H, W), dtype uint8.
            Authentic samples return an empty list.
        """
        if not row["has_mask"]:
            return []

        mask_path = Path(row["mask_paths"][0])
        mask = np.load(mask_path)

        if mask.ndim == 2:
            if mask.shape != (height, width):
                raise ValueError(
                    f"Mask shape {mask.shape} does not match image shape {(height, width)} "
                    f"for case_id={row['case_id']}"
                )
            return [(mask > 0).astype(np.uint8)]

        if mask.ndim == 3:
            if mask.shape[1:] != (height, width):
                raise ValueError(
                    f"Mask spatial shape {mask.shape[1:]} does not match image shape {(height, width)} "
                    f"for case_id={row['case_id']}"
                )

            instance_masks = []
            for i in range(mask.shape[0]):
                instance_masks.append((mask[i] > 0).astype(np.uint8))

            return instance_masks

        raise ValueError(
            f"Unsupported mask shape {mask.shape} for case_id={row['case_id']}"
        )

    def _merge_instance_masks(self, instance_masks, height, width):
        """
        Merge per-instance masks into one semantic mask for training.
        """
        if len(instance_masks) == 0:
            return np.zeros((height, width), dtype=np.uint8)

        merged_mask = np.zeros((height, width), dtype=np.uint8)

        for instance_mask in instance_masks:
            merged_mask = np.logical_or(merged_mask, instance_mask > 0)

        return merged_mask.astype(np.uint8)

    def get_sample_info(self, idx):
        """
        Return original-resolution metadata for evaluation/debugging.
        """
        row = self.df.iloc[idx]

        image = self._load_image(row)
        height, width = image.shape[:2]

        instance_masks = self._load_instance_masks(row, height, width)

        return {
            "case_id": row["case_id"],
            "image_type": row["image_type"],
            "has_mask": row["has_mask"],
            "original_size": (height, width),
            "num_instances": len(instance_masks),
            "instance_masks": instance_masks,
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = self._load_image(row)
        height, width = image.shape[:2]

        instance_masks = self._load_instance_masks(row, height, width)
        mask = self._merge_instance_masks(instance_masks, height, width)

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


if __name__ == "__main__":
    from src.data.load_dataset import load_dataset

    dataset_root = "recodai-luc-scientific-image-forgery-detection"
    df = load_dataset(dataset_root)
    train_dataset = BiomedicalForgeryDataset(df)

    sample_image, sample_mask = train_dataset[1]
    print("Training image shape:", sample_image.shape)
    print("Training mask shape:", sample_mask.shape)

    info = train_dataset.get_sample_info(1)
    print("Case ID:", info["case_id"])
    print("Image type:", info["image_type"])
    print("Original size:", info["original_size"])
    print("Num instances:", info["num_instances"])
    print("Number of instance masks returned:", len(info["instance_masks"]))

    if len(info["instance_masks"]) > 0:
        print("First instance mask shape:", info["instance_masks"][0].shape)