import torch

from src.training.metrics import dice_score_from_logits, iou_score_from_logits


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model.
    dataloader : torch.utils.data.DataLoader
        Training dataloader.
    criterion : callable
        Loss function.
    optimizer : torch.optim.Optimizer
        Optimizer.
    device : torch.device
        Device to run on.

    Returns
    -------
    dict
        Dictionary containing average training loss.
    """
    model.train()

    running_loss = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)

    return {
        "loss": avg_loss,
    }


@torch.no_grad()
def validate_one_epoch(model, dataloader, criterion, device, threshold=0.5):
    """
    Validate the model for one epoch using merged-mask segmentation metrics.

    Parameters
    ----------
    model : torch.nn.Module
        Segmentation model.
    dataloader : torch.utils.data.DataLoader
        Validation dataloader.
    criterion : callable
        Loss function.
    device : torch.device
        Device to run on.
    threshold : float
        Threshold applied after sigmoid for Dice/IoU.

    Returns
    -------
    dict
        Dictionary containing average validation loss, Dice, and IoU.
    """
    model.eval()

    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    num_batches = 0

    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        running_dice += dice_score_from_logits(logits, masks, threshold=threshold)
        running_iou += iou_score_from_logits(logits, masks, threshold=threshold)
        num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    avg_dice = running_dice / max(num_batches, 1)
    avg_iou = running_iou / max(num_batches, 1)

    return {
        "loss": avg_loss,
        "dice": avg_dice,
        "iou": avg_iou,
    }