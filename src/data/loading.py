import torch
import pandas as pd
from pathlib import Path
from tdc.multi_pred import DTI
from torch.utils import data
from torch_geometric import data as pyg_data

from .processing import tokenize_target, tokenize_smiles, smile_to_graph, to_deepsmiles
from .preprocessing import MotifFetcher


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
        mode="deepdta",
        label_to_log=True,
        drug_transform=None,
        target_transform=None,
    ):
        if split not in ["train", "valid", "test"]:
            raise ValueError(
                "Invalid split type. Expected one of: ['train', 'valid', 'test']"
            )

        modes = ["deepdta", "graphdta", "widedta"]
        if mode.lower() not in modes:
            raise ValueError(f'The mode must be: {", ".join(modes)}')

        self.name = name.upper()
        self.path = Path(path)
        self.data = DTI(name=name, path=path)
        self.mode = mode.lower()

        if label_to_log:
            self.data.convert_to_log()

        self.data = self.data.get_split()[split]

        if mode == "widedta":
            self._load_motifs()

        self.drug_transform = drug_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def _load_motifs(self):
        mf = MotifFetcher()
        motifs = mf.get_motifs(self.data, self.path, self.name)
        self.data = pd.merge(self.data, motifs, on="Target_ID", how="left")

        return

    def _deepdta(self, drug, target):
        drug = torch.LongTensor(tokenize_smiles(drug))
        target = torch.LongTensor(tokenize_target(target))

        return drug, target

    def _graphdta(self, drug, target, label):
        c_size, features, edge_index = smile_to_graph(drug)
        drug = pyg_data.Data(
            x=torch.Tensor(features),
            edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            y=torch.Tensor([label]),
        )
        drug.__setitem__("c_size", torch.LongTensor([c_size]))
        target = torch.LongTensor(tokenize_target(target))

        return drug, target

    def _widedta(self, drug, target, motif):
        drug = to_deepsmiles(drug)

        drug = torch.LongTensor(tokenize_smiles(drug))
        target = torch.LongTensor(tokenize_target(target))
        motif = torch.LongTensor(tokenize_target(motif))

        return drug, target, motif

    def _process_data(self, index):
        drug, target, label = (
            self.data["Drug"][index],
            self.data["Target"][index],
            self.data["Y"][index],
        )

        label = torch.FloatTensor([label])

        # if self.drug_transform is not None:
        #     self.drug_transform(drug)
        # if self.target_transform is not None:
        #     self.target_transform(target)

        match self.mode:
            case "deepdta":
                drug, target = self._deepdta(drug, target)
                return drug, target, label

            case "graphdta":
                drug, target = self._graphdta(drug, target, label)
                return drug, target, label

            case "widedta":
                motif = self.data["Motif"][index]
                drug, target, motif = self._widedta(drug, target, motif)
                return drug, target, motif, label

    def __getitem__(self, index):
        return self._process_data(index)
