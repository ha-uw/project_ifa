from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from methods.dgraphdta import DGraphDTA
from data.loading import TDCDataset
from data.preprocessing import preprocess_targets
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


if __name__ == "__main__":
    data = TDCDataset("DAVIS")
    preprocess_targets(data)
