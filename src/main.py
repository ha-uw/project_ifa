from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from data.loading import TDCDataset
from data.preprocessing import MotifFetcher
from time import sleep

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

dataset = "DAVIS"
FAST_DEV_RUN = True


def deepdta():
    config_file = r"C:\Users\raulc\code\projeto_ifá\configs\deepdta.yaml"
    deepdta = DeepDTA(config_file, fast_dev_run=FAST_DEV_RUN)
    deepdta.train()


def graphdta(drug_encoder="GCN"):
    config_file = r"C:\Users\raulc\code\projeto_ifá\configs\graphdta.yaml"
    graphdta = GraphDTA(
        config_file, drug_encoder=drug_encoder, fast_dev_run=FAST_DEV_RUN
    )
    graphdta.train()


def make_motifs(dataset):
    counter = 0
    max_attempts = 3
    dataset = TDCDataset(dataset)
    mf = MotifFetcher(concurrent_sessions=3)

    while counter < max_attempts:
        try:
            mf.load_motifs(dataset)
        except Exception as e:
            print(e)
            sleep(60 * 5 * counter)
            counter += 1
        break
    else:
        print("Max number of attempts reached.")


if __name__ == "__main__":
    # datasets = ["KIBA", "DAVIS", "bindingdb_kd", "bindingdb_ki", "bindingdb_ic50"]

    # for ds in datasets:
    #     print(f"\nDoing {ds} ", "-" * 50)
    #     make_motifs(ds)

    # TDCDataset("Davis", mode="widedta")[22]

    # print(torch.__version__)
    # print(torch.cuda.get_device_properties(0))
    # print(torch.cuda.is_available())
    graphdta()
