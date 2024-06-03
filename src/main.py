from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

config_file = r"C:\Users\raulc\code\projeto_ifá\configs\graphdta.yaml"
dataset = "DAVIS"
FAST_DEV_RUN = False


def deepdta():
    deepdta = DeepDTA(config_file, fast_dev_run=FAST_DEV_RUN)
    deepdta.train()


def graphdta(drug_encoder):
    graphdta = GraphDTA(
        config_file, drug_encoder=drug_encoder, fast_dev_run=FAST_DEV_RUN
    )
    graphdta.train()


if __name__ == "__main__":
    graphdta("GAT")
    graphdta("GCN")
    graphdta("GAT_GCN")
    graphdta("GIN")
