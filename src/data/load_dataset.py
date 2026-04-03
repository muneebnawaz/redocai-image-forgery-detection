from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

def _extract_case_id(path):
    return path.stem


def _list_image_files(folder):
    """
    Return all valid image files in a folder, sorted by filename.
    """
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

def _list_npy_files(folder):
    """
    Return all .npy files in a folder, sorted by filename.
    """
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() == ".npy"
        ]
    )

def _build_mask_lookup(mask_folder):
    """
    Build a mapping from case_id to mask path.
    """
    mask_files = _list_npy_files(mask_folder)

    mask_lookup = {}

    for mask_path in mask_files:
        case_id = _extract_case_id(mask_path)
        mask_lookup[case_id] = mask_path

    return mask_lookup

def _build_image_rows(image_folder, image_type, split, mask_lookup=None):
    """
    Build dataset rows from an image folder.
    """
    image_files = _list_image_files(image_folder)
    rows = []

    for image_path in image_files:
        case_id = _extract_case_id(image_path)

        has_mask = False
        num_masks = 0
        mask_paths = []

        if image_type == "forged" and mask_lookup is not None:
            matched_mask = mask_lookup.get(case_id)

            if matched_mask is not None:
                has_mask = True
                num_masks = 1
                mask_paths = [matched_mask]

        rows.append(
            {
                "case_id": case_id,
                "image_path": image_path,
                "mask_paths": mask_paths,
                "num_masks": num_masks,
                "split": split,
                "image_type": image_type,
                "has_mask": has_mask,
            }
        )

    return rows

def load_dataset(dataset_root):
    """
    Load training and supplementary dataset metadata into a dataframe.
    """
    dataset_root = Path(dataset_root)

    train_authentic_folder = dataset_root / "train_images" / "authentic"
    train_forged_folder = dataset_root / "train_images" / "forged"
    train_masks_folder = dataset_root / "train_masks"

    supplementary_folder = dataset_root / "supplemental_images"
    supplementary_masks_folder = dataset_root / "supplemental_masks"

    train_mask_lookup = _build_mask_lookup(train_masks_folder)

    rows = []

    rows.extend(
        _build_image_rows(
            image_folder=train_authentic_folder,
            image_type="authentic",
            split="train",
            mask_lookup=None,
        )
    )

    rows.extend(
        _build_image_rows(
            image_folder=train_forged_folder,
            image_type="forged",
            split="train",
            mask_lookup=train_mask_lookup,
        )
    )

    if supplementary_folder.exists():
        supplementary_mask_lookup = _build_mask_lookup(supplementary_masks_folder)

        rows.extend(
            _build_image_rows(
                image_folder=supplementary_folder,
                image_type="forged",
                split="supplemental",
                mask_lookup=supplementary_mask_lookup,
            )
        )

    df = pd.DataFrame(rows)

    df["_case_id_sort"] = pd.to_numeric(df["case_id"])
    df = df.sort_values(["_case_id_sort", "case_id"]).drop(columns="_case_id_sort")
    df = df.reset_index(drop=True)

    return df


if __name__ == "__main__":
    dataset_root = "recodai-luc-scientific-image-forgery-detection"
    df = load_dataset(dataset_root)
    print(df.head())
    print(df.shape)