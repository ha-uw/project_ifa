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
        path="data",
        label_to_log=False,
        drug_transform=None,
        target_transform=None,
        print_stats=True,
    ):

        self.name = name.lower()
        self.path = Path(path, self.name)
        self.path.parent.mkdir(exist_ok=True, parents=True)

        self.data = DTI(name=self.name, path=self.path, print_stats=print_stats)

        if label_to_log:
            self.data.convert_to_log()

        self.data = self.data.get_data()

        self.drug_transform = drug_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)
