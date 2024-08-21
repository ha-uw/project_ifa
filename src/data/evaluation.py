"""
evaluation.py

Module containing evaluating functions.
"""

import torch


def concordance_index(y, y_pred):
    """
    Optimized calculation of the Concordance Index (CI) using vectorized operations with PyTorch tensors.
    """
    # Compute differences
    y_diffs = torch.unsqueeze(y, 0) - torch.unsqueeze(y, 1) > 0
    y_pred_diffs = torch.unsqueeze(y_pred, 0) - torch.unsqueeze(y_pred, 1)

    # Calculate concordant and tied pairs
    concordant_pairs = torch.sum(y_diffs & (y_pred_diffs > 0))
    tied_pairs = torch.sum(y_diffs & (y_pred_diffs == 0))

    total_pairs = concordant_pairs + 0.5 * tied_pairs
    total_valid_pairs = torch.sum(y_diffs)

    if total_valid_pairs > 0:
        return (total_pairs / total_valid_pairs).item()
    else:
        return 0.0
