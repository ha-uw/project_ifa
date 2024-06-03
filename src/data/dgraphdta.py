import os
import random
import numpy as np
import json
from collections import OrderedDict
import pconsc4


def seq_format(proteins_dic, output_dir):
    for key, value in proteins_dic.items():
        with open(os.path.join(output_dir, f"{key}.fasta"), "w") as f:
            f.write(f">{key}\n{value}\n")


def run_command(bin_path, input_file, output_file, additional_args=""):
    input_file = input_file.replace("(", "\(").replace(")", "\)")
    output_file = output_file.replace("(", "\(").replace(")", "\)")
    cmd = f"{bin_path} {additional_args} -i {input_file} -o {output_file}"
    print(cmd)
    os.system(cmd)


def HHblitsMSA(bin_path, db_path, input_dir, output_dir):
    for fas_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, fas_file)
        output_file = os.path.join(output_dir, fas_file.split(".fasta")[0] + ".hhr")
        output_file_a3m = os.path.join(output_dir, fas_file.split(".fasta")[0] + ".a3m")
        if not os.path.exists(output_file) and not os.path.exists(output_file_a3m):
            additional_args = f"-maxfilt 100000 -realign_max 100000 -d {db_path} -all -B 100000 -Z 100000 -n 3 -e 0.001 -oa3m {output_file_a3m} -cpu 8"
            run_command(bin_path, process_file, output_file, additional_args)


def HHfilter(bin_path, input_dir, output_dir):
    file_prefix = [
        file.split(".a3m")[0] for file in os.listdir(input_dir) if "a3m" in file
    ]
    for msa_file_prefix in file_prefix:
        file_name = f"{msa_file_prefix}.a3m"
        process_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name)
        if not os.path.exists(output_file):
            run_command(bin_path, process_file, output_file, "-id 90")


def reformat(bin_path, input_dir, output_dir):
    for a3m_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, a3m_file)
        output_file = os.path.join(output_dir, a3m_file.split(".a3m")[0] + ".fas")
        if not os.path.exists(output_file):
            run_command(bin_path, process_file, output_file, "-r")


def convertAlignment(bin_path, input_dir, output_dir):
    for fas_file in os.listdir(input_dir):
        process_file = os.path.join(input_dir, fas_file)
        output_file = os.path.join(output_dir, fas_file.split(".fas")[0] + ".aln")
        if not os.path.exists(output_file):
            cmd = f"python {bin_path} {process_file} fasta {output_file}"
            print(cmd)
            os.system(cmd)


def alnFilePrepare():
    print("aln file prepare ...")
    datasets = ["davis", "kiba"]
    for dataset in datasets:
        seq_dir = os.path.join("data", dataset, "seq")
        msa_dir = os.path.join("data", dataset, "msa")
        filter_dir = os.path.join("data", dataset, "hhfilter")
        reformat_dir = os.path.join("data", dataset, "reformat")
        aln_dir = os.path.join("data", dataset, "aln")
        protein_path = os.path.join("data", dataset)
        proteins = json.load(
            open(os.path.join(protein_path, "proteins.txt")),
            object_pairs_hook=OrderedDict,
        )

        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(msa_dir, exist_ok=True)
        os.makedirs(filter_dir, exist_ok=True)
        os.makedirs(reformat_dir, exist_ok=True)
        os.makedirs(aln_dir, exist_ok=True)

        HHblits_bin_path = "..../tool/hhsuite/bin/hhblits"
        HHblits_db_path = "..../dataset/uniclust/uniclust30_2018_08/uniclust30_2018_08"
        HHfilter_bin_path = "..../tool/hhsuite/bin/hhfilter"
        reformat_bin_path = "..../tool/hhsuite/scripts/reformat.pl"
        convertAlignment_bin_path = "..../tool/CCMpred/scripts/convert_alignment.py"

        seq_format(proteins, seq_dir)
        HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
        HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
        reformat(reformat_bin_path, filter_dir, reformat_dir)
        convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)

        print("aln file prepare over.")


def pconsc4Prediction():
    datasets = ["davis", "kiba"]
    model = pconsc4.get_pconsc4()
    for dataset in datasets:
        aln_dir = os.path.join("data", dataset, "hhfilter")
        output_dir = os.path.join("data", dataset, "pconsc4")
        os.makedirs(output_dir, exist_ok=True)
        file_list = os.listdir(aln_dir)
        random.shuffle(file_list)
        for file in file_list:
            input_file = os.path.join(aln_dir, file)
            output_file = os.path.join(output_dir, file.split(".a3m")[0] + ".npy")
            if not os.path.exists(output_file):
                try:
                    print("process", input_file)
                    pred = pconsc4.predict(model, input_file)
                    np.save(output_file, pred["cmap"])
                    print(output_file, "over.")
                except:
                    print(output_file, "error.")


if __name__ == "__main__":
    alnFilePrepare()
    pconsc4Prediction()
