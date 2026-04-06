from src.data.load_dataset import load_dataset
from src.data.dataset import BiomedicalForgeryDataset
from scripts.visualize_sample import visualize_sample


def print_sample_info(dataset, idx):
    row = dataset.df.iloc[idx]
    image, mask = dataset[idx]

    print("=" * 60)
    print(f"Index      : {idx}")
    print(f"Case ID    : {row['case_id']}")
    print(f"Image type : {row['image_type']}")
    print(f"Has mask   : {row['has_mask']}")
    print(f"Image shape: {tuple(image.shape)}")
    print(f"Mask shape : {tuple(mask.shape)}")
    print(f"Mask sum   : {mask.sum().item()}")
    print(f"Mask unique: {mask.unique()}")


if __name__ == "__main__":
    df = load_dataset("recodai-luc-scientific-image-forgery-detection")

    dataset = BiomedicalForgeryDataset(
        df,
        target_size=(512, 512),
        transforms=None
    )

    # Pick some authentic and forged samples
    authentic_indices = df[df["image_type"] == "authentic"].index[123:130].tolist()
    forged_indices = df[df["image_type"] == "forged"].index[123:130].tolist()
    sample_indices = authentic_indices + forged_indices

    for idx in sample_indices:
        print_sample_info(dataset, idx)
        visualize_sample(dataset, idx)