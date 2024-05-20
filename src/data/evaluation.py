import numpy as np
from math import sqrt
from scipy import stats
from lifelines.utils import concordance_index as ci


# TODO I probably can get these dutring train. check check check
def rmse(y, y_pred):
    rmse = sqrt(((y - y_pred) ** 2).mean(axis=0))
    return rmse


def mse(y, y_pred):
    mse = ((y - y_pred) ** 2).mean(axis=0)
    return mse


def pearson(y, y_pred):
    rp = np.corrcoef(y, y_pred)[0, 1]
    return rp


def spearman(y, y_pred):
    rs = stats.spearmanr(y, y_pred)[0]
    return rs


def concordance_index(y, y_pred):
    """
    Calculate the concordance index (CI) between the true labels (y) and the predicted labels (y_pred).

    The concordance index is a measure of how well the predicted labels rank the samples compared to the true labels.
    It ranges from 0 to 1, where 0 indicates no concordance and 1 indicates perfect concordance.

    Parameters:
        y (array-like): The true labels.
        y_pred (array-like): The predicted labels.

    Returns:
        float: The concordance index.

    """
    concordance_index = ci(y, y_pred)

    return concordance_index
