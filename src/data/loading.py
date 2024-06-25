import pandas as pd
from pathlib import Path
from tdc.multi_pred import DTI
from torch.utils import data


class TDCDataset(data.Dataset):
    """
    A custom dataset for loading and processing original TDC data, which is used as input data in DeepDTA model.

    Args:
         name (str): TDC dataset name.
         split (str): Data split type (train, valid or test).
         path (str): dataset download/local load path (default: "./data")
         drug_transform: Transform operation (default: None)
         target_transform: Transform operation (default: None)
         y_log (bool): Whether convert y values to log space. (default: True)
    """

    data: pd.DataFrame

    def __init__(
        self,
        name: str,
        split="train",
        path="./data",
        label_to_log=True,
        drug_transform=None,
        target_transform=None,
    ):
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "Invalid split type. Expected one of: ['train', 'valid', 'test']"
            )

        self.name = name.upper()
        self.path = Path(path)
        self.data = DTI(name=name, path=path)

        if label_to_log:
            self.data.convert_to_log()

        self.data = self.data.get_split()[split]

        self.drug_transform = drug_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
