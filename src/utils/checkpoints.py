import torch


def save_checkpoint(path, model, optimizer, epoch, best_val_score):
    """
    Save a training checkpoint.

    Parameters
    ----------
    path : str
        Output checkpoint path.
    model : torch.nn.Module
        Model to save.
    optimizer : torch.optim.Optimizer
        Optimizer to save.
    epoch : int
        Current epoch.
    best_val_score : float
        Best validation score so far.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_score": best_val_score,
    }
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    """
    Load a training checkpoint.

    Parameters
    ----------
    path : str
        Checkpoint path.
    model : torch.nn.Module
        Model to load into.
    optimizer : torch.optim.Optimizer or None
        Optimizer to load into if provided.
    map_location : str
        Device mapping.

    Returns
    -------
    dict
        Loaded checkpoint dictionary.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint