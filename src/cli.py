import argparse
from methods.widedta import WideDTA
from methods.deepdta import DeepDTA
from methods.graphdta import GraphDTA
from pathlib import Path
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description="Run methods with specified action")
    parser.add_argument(
        "--method",
        type=str,
        choices=["widedta", "deepdta", "graphdta"],
        required=True,
        help="The method to run",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["davis", "kiba", "bindingdb_ki", "bindingdb_kd", "bindingdb_ic50"],
        required=True,
        help="The dataset to train",
    )
    parser.add_argument(
        "action",
        type=str,
        choices=["run_5_fold"],
        help="The action to perform",
    )

    return parser.parse_args()


def main():
    args = parse_cmd_line_arguments()

    # Initialize the selected method
    if args.method == "widedta":
        method_instance = WideDTA(
            config_file=Path("configs", "WideDTA", f"{args.dataset}.yaml"),
            fast_dev_run=False,
        )
    elif args.method == "deepdta":
        method_instance = DeepDTA(
            config_file=Path("configs", "DeepDTA", f"{args.dataset}.yaml"),
            fast_dev_run=False,
        )
    elif args.method == "graphdta":
        method_instance = GraphDTA(
            config_file=Path("configs", "GraphDTA", f"{args.dataset}.yaml")
        )

    if args.action == "run_5_fold":
        method_instance.run_k_fold_validation(n_splits=5)
    # elif args.action == "resume_training":
    #     # Assuming each method has a resume_training method for this example
    #     method_instance.resume_training()
