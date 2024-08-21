"""
loading.py

Module for loading and handling datasets.
"""

import pandas as pd
from pathlib import Path
from tdc.multi_pred import DTI
from torch.utils.data import Dataset


class TDCDataset(Dataset):
    """
    A dataset class for TDC (Therapeutic Data Commons) dataset.
    Args:
        name (str): The name of the dataset.
        path (str, optional): The path to the dataset. Defaults to "data".
        label_to_log (bool, optional): Whether to convert labels to logarithmic scale. Defaults to False.
        print_stats (bool, optional): Whether to print dataset statistics. Defaults to True.
        split (str, optional): The split of the dataset to use. Defaults to None.
        harmonize_affinities (bool, optional): Whether to harmonize affinities. Defaults to False.
    Attributes:
        name (str): The name of the dataset.
        path (Path): The path to the dataset.
        data (pd.DataFrame): The dataset.
    """

    ...

    name: str
    path: Path
    data: pd.DataFrame

    def __init__(
        self,
        name: str,
        path="data",
        label_to_log=False,
        print_stats=True,
        split=None,
        harmonize_affinities=False,
    ):

        self.name = name.lower()
        self.path = Path(path, self.name)
        self.path.parent.mkdir(exist_ok=True, parents=True)

        self.data = DTI(name=self.name, path=self.path, print_stats=print_stats)

        if harmonize_affinities:
            self.data.harmonize_affinities(mode="mean")

        if label_to_log:
            self.data.convert_to_log()

        if split:
            self.data = self.data.get_split()[split]
        else:
            self.data = self.data.get_data()

    def __len__(self):
        return len(self.data)
