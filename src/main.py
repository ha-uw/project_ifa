from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

config_file = r"C:\Users\raulc\code\projeto_ifá\configs\graphdta.yaml"


def deepdta():
    deepdta = DeepDTA(config_file)
    deepdta.train()


def graphdta():
    graphdta = GraphDTA(config_file)
    graphdta.train()


if __name__ == "__main__":
    graphdta()
