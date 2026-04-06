from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.load_dataset import load_dataset
from src.data.dataset import BiomedicalForgeryDataset
from src.models.unet_baseline import UNet


def main():
    df = load_dataset("recodai-luc-scientific-image-forgery-detection")

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["image_type"]
    )

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = BiomedicalForgeryDataset(
        df=train_df,
        target_size=(512, 512)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    images, masks = next(iter(train_loader))

    model = UNet(in_channels=3, out_channels=1)

    outputs = model(images)

    print("Input image shape:", images.shape)
    print("Target mask shape:", masks.shape)
    print("Model output shape:", outputs.shape)
    print("Output dtype:", outputs.dtype)
    print("Output min:", outputs.min().item())
    print("Output max:", outputs.max().item())


if __name__ == "__main__":
    main()