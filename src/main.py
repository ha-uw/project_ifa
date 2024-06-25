from methods.widedta import WideDTA
from methods.deepdta import DeepDTA

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

# CUDA_LAUNCH_BLOCKING = 1
if __name__ == "__main__":

    wd = WideDTA(
        config_file=r"C:\Users\raulc\code\projeto_ifá\configs\widedta.yaml",
        fast_dev_run=True,
    )

    wd.train()
