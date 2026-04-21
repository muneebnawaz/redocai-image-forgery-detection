import torch


def dice_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    """
    Compute mean Dice score for a batch from raw logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape (B, 1, H, W).
    targets : torch.Tensor
        Ground-truth binary masks of shape (B, 1, H, W).
    threshold : float
        Threshold applied after sigmoid.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean Dice score across the batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    union = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score_from_logits(logits, targets, threshold=0.5, eps=1e-7):
    """
    Compute mean IoU score for a batch from raw logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs of shape (B, 1, H, W).
    targets : torch.Tensor
        Ground-truth binary masks of shape (B, 1, H, W).
    threshold : float
        Threshold applied after sigmoid.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    float
        Mean IoU score across the batch.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    dims = (1, 2, 3)
    intersection = (preds * targets).sum(dim=dims)
    total = preds.sum(dim=dims) + targets.sum(dim=dims)
    union = total - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()