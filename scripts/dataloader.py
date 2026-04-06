from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.data.load_dataset import load_dataset
from src.data.dataset import BiomedicalForgeryDataset

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
    df=train_df
)

val_dataset = BiomedicalForgeryDataset(
    df=val_df
)

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0,
    pin_memory=False
)

batch = next(iter(train_loader))
images, masks = batch

print("Images shape:", images.shape)
print("Masks shape:", masks.shape)
print("Mask unique values:", masks.unique())

mask_sums = masks.sum(dim=(1, 2, 3))
print("Mask pixel sums per sample:", mask_sums)
print("Samples with non-empty masks:", (mask_sums > 0).sum().item())