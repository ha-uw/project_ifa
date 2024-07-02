from methods.widedta import WideDTA
from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from pathlib import Path

from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")

# CUDA_LAUNCH_BLOCKING = 1
if __name__ == "__main__":

    wd = WideDTA(
        config_file=Path("configs", "WideDTA", "bindingdb_ki.yaml"),
        fast_dev_run=False,
    )

    dd = DeepDTA(
        config_file=Path("configs", "DeepDTA", "davis.yaml"),
        fast_dev_run=False,
    )

    gd = GraphDTA(config_file=Path("configs", "GraphDTA", "davis.yaml"))

    wd.run_k_fold_validation(5)
