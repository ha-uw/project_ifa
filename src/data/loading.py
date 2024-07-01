import pandas as pd
from pathlib import Path
from tdc.multi_pred import DTI
from torch.utils.data import Dataset


class TDCDataset(Dataset):
    """ """

    name: str
    path: Path
    data: pd.DataFrame

    def __init__(
        self,
        name: str,
        path="data",
        label_to_log=False,
        print_stats=True,
    ):

        self.name = name.lower()
        self.path = Path(path, self.name)
        self.path.parent.mkdir(exist_ok=True, parents=True)

        self.data = DTI(name=self.name, path=self.path, print_stats=print_stats)

        if label_to_log:
            self.data.convert_to_log()

        self.data = self.data.get_data()

    def __len__(self):
        return len(self.data)
