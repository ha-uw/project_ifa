import argparse
from src.methods.widedta import WideDTA
from src.methods.deepdta import DeepDTA
from src.methods.graphdta import GraphDTA
from pathlib import Path
from torch import set_float32_matmul_precision

set_float32_matmul_precision("medium")


def parse_cmd_line_arguments():
    parser = argparse.ArgumentParser(description="Your application description")

    # Define the 'action' as a positional argument
    parser.add_argument(
        "action",
        type=str,
        choices=["run_5_fold", "resume"],
        help="The action to perform",
    )

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
        "--last_fold",
        type=int,
        required=False,
        help="The last completed fold",
    )
    parser.add_argument(
        "--version",
        type=int,
        required=False,
        help="The version of the training to resume",
    )

    # Return the parsed arguments
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
            config_file=Path("configs", "GraphDTA", f"{args.dataset}.yaml"),
            drug_encoder="GAT_GCN",
        )

    if args.action == "run_5_fold":
        method_instance.run_k_fold_validation(n_splits=5)
    elif args.action == "resume":
        # Assuming each method has a resume_training method for this example
        method_instance.resume_training(
            version=args.version, last_completed_fold=args.last_fold
        )
