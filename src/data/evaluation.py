import torch

import numpy as np
from math import sqrt
from scipy import stats


# # TODO I probably can get these dutring train. check check check
def rmse(y, y_pred):
    rmse = sqrt(((y - y_pred) ** 2).mean(axis=0))
    return rmse


def mse(y, y_pred):
    mse = ((y - y_pred) ** 2).mean(axis=0)
    return mse


# def pearson(y, y_pred):
#     rp = np.corrcoef(y, y_pred)[0, 1]
#     return rp


# def spearman(y, y_pred):
#     rs = stats.spearmanr(y, y_pred)[0]
#     return rs


def concordance_index_ORIGINAL(y, y_pred):
    """
    Calculate the Concordance Index (CI), which is a metric to measure the proportion of `concordant pairs
    <https://en.wikipedia.org/wiki/Concordant_pair>`_ between real and
    predict values.

    Args:
        y (array): real values.
        y_pred (array): predicted values.
    """
    total_loss = 0
    pair = 0
    for i in range(1, len(y)):
        for j in range(0, i):
            if i is not j:
                if y[i] > y[j]:
                    pair += 1
                    total_loss += 1 * (y_pred[i] > y_pred[j]) + 0.5 * (
                        y_pred[i] == y_pred[j]
                    )

    if pair:
        result = total_loss / pair
        return result.item()
    else:
        return 0.0


# def concordance_index(y, y_pred):
#     """
#     Optimized calculation of the Concordance Index (CI) using vectorized operations.
#     """
#     y = np.array(y)
#     y_pred = np.array(y_pred)

#     y_diffs = np.subtract.outer(y, y) > 0
#     y_pred_diffs = np.subtract.outer(y_pred, y_pred)

#     concordant_pairs = np.sum((y_diffs & (y_pred_diffs > 0)).flatten())
#     tied_pairs = np.sum((y_diffs & (y_pred_diffs == 0)).flatten())

#     total_pairs = concordant_pairs + 0.5 * tied_pairs
#     total_valid_pairs = np.sum(y_diffs)

#     if total_valid_pairs > 0:
#         return total_pairs / total_valid_pairs
#     else:
#         return 0.0


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
