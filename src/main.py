from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from methods.dgraphdta import DGraphDTA
from tdc.multi_pred import DTI
from data.loading import TDCDataset
from data.preprocessing import MotifFetcher
from time import sleep

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

config_file = r"C:\Users\raulc\code\projeto_ifá\configs\dgraphdta.yaml"
dataset = "DAVIS"
FAST_DEV_RUN = True


def deepdta():
    deepdta = DeepDTA(config_file, fast_dev_run=FAST_DEV_RUN)
    deepdta.train()


def graphdta(drug_encoder):
    graphdta = GraphDTA(
        config_file, drug_encoder=drug_encoder, fast_dev_run=FAST_DEV_RUN
    )
    graphdta.train()


def dgraphdta():
    dgraphdta = DGraphDTA(config_file, fast_dev_run=FAST_DEV_RUN)
    dgraphdta.train()


def do_motif(dataset):
    dataset = TDCDataset(dataset)
    mf = MotifFetcher(concurrent_sessions=3)
    mf.load_motif_file(dataset)


if __name__ == "__main__":
    counter = 0
    max_attempts = 3
    datasets = ["KIBA"]

    while counter < max_attempts:
        for ds in datasets:
            try:
                do_motif(ds)
                datasets.remove(ds)
            except Exception as e:
                print(e)
                sleep(60 * 5 * counter)
                counter += 1
