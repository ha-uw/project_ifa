from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from data.loading import TDCDataset
from data.preprocessing import MotifFetcher
from time import sleep

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")
