import torch

# import numpy as np
# from math import sqrt
# from scipy import stats


# # TODO I probably can get these dutring train. check check check
# def rmse(y, y_pred):
#     rmse = sqrt(((y - y_pred) ** 2).mean(axis=0))
#     return rmse


# def mse(y, y_pred):
#     mse = ((y - y_pred) ** 2).mean(axis=0)
#     return mse


# def pearson(y, y_pred):
#     rp = np.corrcoef(y, y_pred)[0, 1]
#     return rp


# def spearman(y, y_pred):
#     rs = stats.spearmanr(y, y_pred)[0]
#     return rs


# def concordance_index(y, y_pred):
#     """
#     Calculate the concordance index (ci) between true labels and predicted scores using vectorized operations.

#     Args:
#         y (torch.Tensor): True labels.
#         y_pred (torch.Tensor): Predicted scores.

#     Returns:
#         float: Concordance index.
#     """
#     # Ensure the inputs are tensors
#     if not isinstance(y, torch.Tensor):
#         y = torch.tensor(y, dtype=torch.float32)
#     if not isinstance(y_pred, torch.Tensor):
#         y_pred = torch.tensor(y_pred, dtype=torch.float32)

#     # Ensure y and y_pred are 1-dimensional
#     y = y.view(-1)
#     y_pred = y_pred.view(-1)

#     # Create pairwise comparison matrices
#     diff_true = y.unsqueeze(0) - y.unsqueeze(1)
#     diff_pred = y_pred.unsqueeze(0) - y_pred.unsqueeze(1)

#     # Mask to ignore pairs with the same true values
#     non_equal_mask = diff_true != 0

#     # Calculate concordant pairs
#     concordant = ((diff_pred > 0) == (diff_true > 0)).float()
#     concordant.add_((diff_pred == 0).float() * 0.5)  # Handling ties in predictions

#     # Apply the non_equal_mask
#     concordant.mul_(non_equal_mask.float())

#     # Count permissible pairs and concordant pairs
#     permissible = non_equal_mask.float().sum().item()
#     concordant = concordant.sum().item()

#     # Avoid zero division
#     if permissible == 0:
#         return 0.0

#     # Calculate and return ci
#     return concordant / permissible


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
