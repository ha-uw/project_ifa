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


def concordance_index(y, y_pred):
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
