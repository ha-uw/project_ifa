import os
import random
import numpy as np
import json
import conkit
from collections import OrderedDict


import subprocess


def write_to_file(output_dir, filename, content):
    """Function to write content to a file."""
    with open(os.path.join(output_dir, f"{filename}.fasta"), "w") as f:
        f.write(content)


def seq_format(proteins_dic, output_dir):
    for key, value in proteins_dic.items():
        content = f">{key}\n{value}\n"
        write_to_file(output_dir, key, content)
        print(
            f"Protein sequence for {key} has been written to {output_dir}/{key}.fasta"
        )


# Function to run a command in the terminal
def run_command(bin_path, input_file, output_file, additional_args=""):
    # Construct the command
    cmd = f"{bin_path} {additional_args} -i {input_file} -o {output_file}"

    # Use subprocess.run for better error handling
    try:
        subprocess.run(cmd, check=True, shell=True)
        print(f"Command '{cmd}' executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Command '{cmd}' failed with error code: {e.returncode}")


def process_files(bin_path, input_dir, output_dir, input_ext, output_ext, command_args):
    """Generic function to process files from one format to another."""
    for input_file in os.listdir(input_dir):
        if input_file.endswith(input_ext):
            process_file = os.path.join(input_dir, input_file)
            output_file = os.path.join(
                output_dir, input_file.replace(input_ext, output_ext)
            )

            # Run the command only if the output file does not exist
            if not os.path.exists(output_file):
                additional_args = command_args.format(
                    input_file=process_file, output_file=output_file
                )
                run_command(bin_path, additional_args)


# Function to run HHblits for Multiple Sequence Alignment (MSA)
def HHblitsMSA(bin_path, db_path, input_dir, output_dir):
    command_args = f"-maxfilt 100000 -realign_max 100000 -d {db_path} -all -B 100000 -Z 100000 -n 3 -e 0.001 -oa3m {{output_file}} -cpu 8"
    process_files(bin_path, input_dir, output_dir, ".fasta", ".hhr", command_args)


# Function to filter the MSA files
def HHfilter(bin_path, input_dir, output_dir):
    command_args = "-id 90"
    process_files(bin_path, input_dir, output_dir, ".a3m", ".a3m", command_args)


# Function to reformat the .a3m files
def reformat(bin_path, input_dir, output_dir):
    command_args = "-r a3m fas {input_file} {output_file}"
    process_files(bin_path, input_dir, output_dir, ".a3m", ".fasta", command_args)


# Function to convert the alignment files
def convertAlignment(bin_path, input_dir, output_dir):
    command_args = "-in {input_file} -out {output_file} -output clustal"
    process_files(bin_path, input_dir, output_dir, ".fasta", ".aln", command_args)


def prepare_alignment_files():
    print("Preparing alignment files...")
    datasets = ["davis", "kiba"]

    # Define the paths to the tools
    HHblits_bin_path = "..../tool/hhsuite/bin/hhblits"
    HHblits_db_path = "..../dataset/uniclust/uniclust30_2018_08/uniclust30_2018_08"
    HHfilter_bin_path = "..../tool/hhsuite/bin/hhfilter"
    reformat_bin_path = "..../tool/hhsuite/scripts/reformat.pl"
    convertAlignment_bin_path = "..../tool/CCMpred/scripts/convert_alignment.py"

    for dataset in datasets:
        # Define the directories for each step of the process
        seq_dir = os.path.join("data", dataset, "seq")
        msa_dir = os.path.join("data", dataset, "msa")
        filter_dir = os.path.join("data", dataset, "hhfilter")
        reformat_dir = os.path.join("data", dataset, "reformat")
        aln_dir = os.path.join("data", dataset, "aln")

        # Load the protein sequences
        protein_path = os.path.join("data", dataset)
        proteins = json.load(
            open(os.path.join(protein_path, "proteins.txt")),
            object_pairs_hook=OrderedDict,
        )

        # Create the directories if they don't exist
        os.makedirs(seq_dir, exist_ok=True)
        os.makedirs(msa_dir, exist_ok=True)
        os.makedirs(filter_dir, exist_ok=True)
        os.makedirs(reformat_dir, exist_ok=True)
        os.makedirs(aln_dir, exist_ok=True)

        # Process the protein sequences
        seq_format(proteins, seq_dir)
        HHblitsMSA(HHblits_bin_path, HHblits_db_path, seq_dir, msa_dir)
        HHfilter(HHfilter_bin_path, msa_dir, filter_dir)
        reformat(reformat_bin_path, filter_dir, reformat_dir)
        convertAlignment(convertAlignment_bin_path, reformat_dir, aln_dir)

    print("Alignment file preparation complete.")


def predict_protein_contacts():
    """Predict protein contacts using the pconsc4 model."""
    # Define the datasets
    datasets = ["davis", "kiba"]

    # Load the pconsc4 model
    model = pconsc4.get_pconsc4()

    # Iterate over the datasets
    for dataset in datasets:
        # Define the directories
        aln_dir = os.path.join("data", dataset, "hhfilter")
        output_dir = os.path.join("data", dataset, "pconsc4")

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Get a list of files in the alignment directory and shuffle it
        file_list = os.listdir(aln_dir)
        random.shuffle(file_list)

        # Iterate over the files in the shuffled list
        for file in file_list:
            # Define the input and output files
            input_file = os.path.join(aln_dir, file)
            output_file = os.path.join(output_dir, file.split(".a3m")[0] + ".npy")

            # If the output file doesn't exist, try to predict the protein contacts
            if not os.path.exists(output_file):
                try:
                    print(f"Processing {input_file}...")
                    pred = pconsc4.predict(model, input_file)
                    np.save(output_file, pred["cmap"])
                    print(f"Saved prediction to {output_file}.")
                except:
                    print(f"Error processing {output_file}.")
