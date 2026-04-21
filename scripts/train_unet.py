from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.load_dataset import load_dataset
from src.data.dataset import BiomedicalForgeryDataset
from src.models.unet_baseline import UNet
from src.losses.segmentation_losses import BCEDiceLoss
from src.training.engine import train_one_epoch, validate_one_epoch
from src.utils.checkpoints import save_checkpoint


def main():

    # -------------------------
    # Config
    # -------------------------
    data_root = "recodai-luc-scientific-image-forgery-detection"
    batch_size = 4
    num_workers = 0
    num_epochs = 20
    learning_rate = 1e-3
    val_fraction = 0.2
    threshold = 0.5

    debug = True
    debug_num_samples = 32

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Build dataframe
    # -------------------------
    df = load_dataset(data_root)

    if debug:
        df = df.sample(n=min(debug_num_samples, len(df)), random_state=42).reset_index(drop=True)
        print(f"Debug mode: using {len(df)} samples")

    # -------------------------
    # Dataset
    # -------------------------
    dataset = BiomedicalForgeryDataset(
        df=df,
        target_size=(512, 512),
    )

    # -------------------------
    # Train / val split
    # -------------------------
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # -------------------------
    # Dataloaders
    # -------------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # -------------------------
    # Model, loss, optimizer
    # -------------------------
    model = UNet(in_channels=3, out_channels=1)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # -------------------------
    # Training loop
    # -------------------------
    best_val_dice = -1.0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = validate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=threshold,
        )

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_dice={val_metrics['dice']:.4f} | "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            save_checkpoint(
                path=checkpoint_dir / "best_model.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_score=best_val_dice,
            )
            print(f"Saved new best model with val_dice={best_val_dice:.4f}")


if __name__ == "__main__":
    main()