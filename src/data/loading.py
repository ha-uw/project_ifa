import torch
from tdc.multi_pred import DTI
from torch.utils import data
from torch_geometric import data as pyg_data

from .processing import tokenize_target, tokenize_smiles, smile_to_graph


class TDCDataset(data.Dataset):
    """
    A custom dataset for loading and processing original TDC data, which is used as input data in DeepDTA model.

    Args:
         name (str): TDC dataset name.
         split (str): Data split type (train, valid or test).
         path (str): dataset download/local load path (default: "./data")
         mode (str): encoding mode (default: cnn_cnn)
         drug_transform: Transform operation (default: None)
         target_transform: Transform operation (default: None)
         y_log (bool): Whether convert y values to log space. (default: True)
    """

    def __init__(
        self,
        name: str,
        split="train",
        path="./data",
        mode_drug="cnn",
        mode_target="cnn",
        y_to_log=True,
        drug_transform=None,
        target_transform=None,
    ):
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "Invalid split type. Expected one of: ['train', 'valid', 'test']"
            )
        self.data = DTI(name=name, path=path)
        self.mode_drug = mode_drug.lower()
        self.mode_target = mode_target.lower()
        if y_to_log:
            self.data.convert_to_log()
        self.data = self.data.get_split()[split]
        self.drug_transform = drug_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        drug, target, label = (
            self.data["Drug"][idx],
            self.data["Target"][idx],
            self.data["Y"][idx],
        )

        # Drug
        if self.mode_drug == "cnn":
            drug = torch.LongTensor(tokenize_smiles(drug))

        elif self.mode_drug == "gcn":
            c_size, features, edge_index = smile_to_graph(drug)
            drug = pyg_data.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.Tensor([label]),
            )
            drug.__setitem__("c_size", torch.LongTensor([c_size]))

        # Target
        if self.mode_target == "cnn":
            target = torch.LongTensor(tokenize_target(target))

        # Label
        label = torch.FloatTensor([label])

        if self.drug_transform is not None:
            self.drug_transform(drug)
        if self.target_transform is not None:
            self.target_transform(target)
        return drug, target, label
