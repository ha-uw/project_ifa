import os
import subprocess
from pathlib import Path

from .loading import TDCDataset
import conkit.io


class ContactMap:
    HHBLIST_BIN: Path
    HHBLIST_DB: Path
    HHFILTER_BIN: Path

    def __init__(
        self,
        hhblist_bin: str | Path,
        hhfilter_bin: str | Path,
        hhblist_db_path: str | Path,
    ):

        if not isinstance(hhblist_bin, Path):
            self.HHBLIST_BIN = Path(hhblist_bin).as_posix()
        if not isinstance(hhfilter_bin, Path):
            self.HHFILTER_BIN = Path(hhfilter_bin).as_posix()
        if not isinstance(hhblist_db_path, Path):
            self.HHBLIST_DB = Path(hhblist_db_path).as_posix()

    # Function to run a command in the terminal
    def _run(self, bin_path, input_file, output_file, arguments=""):
        cmd = f"{bin_path} {arguments} -i {input_file} -o {output_file}"

        try:
            subprocess.run(cmd, check=True, shell=False)
            print(f"Command '{cmd}' executed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Command '{cmd}' failed with error code: {e.returncode}")

    def multiple_seq_alignment(
        self,
        input_file,
        output_dir,
        max_filter=100000,
        realign_max=100000,
        block_size=100000,
        max_lines_in_summary=100000,
        iterations=3,
        e_value_th=0.001,
        cpu_threads=8,
    ):
        input_file = Path(input_file).as_posix()
        naked_name = Path(input_file).stem
        output_file = Path(output_dir, f"{naked_name}.a3m").as_posix()
        command_args = f"-maxfilt {max_filter} -realign_max {realign_max} -d {self.HHBLIST_DB} -B {block_size} -Z {max_lines_in_summary} -n {iterations} -e {e_value_th} -oa3m {output_file} -cpu {cpu_threads}"
        self._run(
            self.HHBLIST_BIN,
            input_file=input_file,
            output_file=output_file,
            arguments=command_args,
        )

    def filter_alignment(self, input_file, output_dir, id_threshold=90):
        naked_name = Path(input_file).stem
        output_file = Path(output_dir, f"{naked_name}_id_{id_threshold}.a3m")
        command_args = f"-id {id_threshold}"

        self._run(
            self.HHFILTER_BIN,
            input_file=input_file,
            output_file=output_file,
            arguments=command_args,
        )

    # Function to reformat the .a3m files
    def reformat(self, input_file, output_dir):
        naked_name = Path(input_file).stem
        output_file = Path(output_dir, f"{naked_name}.fas")

        conkit.io.convert(input_file, "a3m", output_file, "fas")

    # Function to convert the alignment files
    def convert_alignment(self, input_file, output_dir):
        # conkit
        command_args = "-in {input_file} -out {output_file} -output clustal"

        return command_args

    def write_to_fasta(self, output_dir, id, sequence):
        f_path = Path(output_dir, f"{id}.fasta")
        if not f_path.is_file():
            with open(f_path, "w") as f:
                f.write(f">{id}\n{sequence}")
            return True
        else:
            return False

    def df_to_fasta(self, df, output_dir: str):
        """ """
        # Filter the DataFrame
        filtered_df = df[["Target_ID", "Target"]]
        counter = 0

        for _, row in filtered_df.iterrows():
            if self.write_to_fasta(
                output_dir, id=row["Target_ID"], sequence=row["Target"]
            ):
                counter += 1
            else:
                continue

        return counter

    def preprocess_targets(self, dataset: TDCDataset):
        # Define the directories for each step of the process
        fasta_dir = Path("data", dataset.name, "fasta")
        msa_dir = Path("data", dataset.name, "msa")
        filter_dir = Path("data", dataset.name, "hhfilter")
        reformat_dir = Path("data", dataset.name, "reformat")
        aln_dir = Path("data", dataset.name, "alignment")

        # Create the directories if they don't exist
        os.makedirs(fasta_dir, exist_ok=True)
        os.makedirs(msa_dir, exist_ok=True)
        os.makedirs(filter_dir, exist_ok=True)
        os.makedirs(reformat_dir, exist_ok=True)
        os.makedirs(aln_dir, exist_ok=True)

        # Process the target sequences
        n_fasta_created = self.df_to_fasta(dataset.data, output_dir=fasta_dir)
        print(f"{n_fasta_created} fasta files created.")

        for f in Path(fasta_dir).glob("*.fasta"):
            self.multiple_seq_alignment(f, output_dir=msa_dir)
            break

        for f in Path(msa_dir).glob("*.a3m"):
            self.filter_alignment(f, output_dir=filter_dir)

        for f in Path(filter_dir).glob("*.a3m"):
            self.reformat(
                f,
            )

        self.multiple_seq_alignment(
            HHblits_bin_path, HHblits_db_path, fasta_dir, msa_dir
        )
        self.filter_alignment(HHfilter_bin_path, msa_dir, filter_dir)
        self.reformat(reformat_bin_path, filter_dir, reformat_dir)
        self.convert_alignment(convertAlignment_bin_path, reformat_dir, aln_dir)


# -------------------------------------------------------------------------------
# def predict_protein_contacts():
#     """Predict protein contacts using the pconsc4 model."""
#     # Define the datasets
#     datasets = ["davis", "kiba"]

#     # Load the pconsc4 model
#     model = pconsc4.get_pconsc4()

#     # Iterate over the datasets
#     for dataset in datasets:
#         # Define the directories
#         aln_dir = os.path.join("data", dataset, "hhfilter")
#         output_dir = os.path.join("data", dataset, "pconsc4")

#         # Create the output directory if it doesn't exist
#         os.makedirs(output_dir, exist_ok=True)

#         # Get a list of files in the alignment directory and shuffle it
#         file_list = os.listdir(aln_dir)
#         random.shuffle(file_list)

#         # Iterate over the files in the shuffled list
#         for file in file_list:
#             # Define the input and output files
#             input_file = os.path.join(aln_dir, file)
#             output_file = os.path.join(output_dir, file.split(".a3m")[0] + ".npy")

#             # If the output file doesn't exist, try to predict the protein contacts
#             if not os.path.exists(output_file):
#                 try:
#                     print(f"Processing {input_file}...")
#                     pred = pconsc4.predict(model, input_file)
#                     np.save(output_file, pred["cmap"])
#                     print(f"Saved prediction to {output_file}.")
#                 except:
#                     print(f"Error processing {output_file}.")
