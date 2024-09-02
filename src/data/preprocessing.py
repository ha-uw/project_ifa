"""
preprocessing.py

Module for preprocessing data.
"""

import aiofiles
import asyncio
import aiohttp
from bs4 import BeautifulSoup

from pathlib import Path
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset


class MotifFetcher:
    """
    A class for fetching motifs from a given protein sequence.

    Args:
        concurrent_sessions (int): The number of concurrent HTTP sessions to use for fetching motifs. Default is 2.
        max_unsuccessful_responses (int): The maximum number of unsuccessful HTTP responses allowed before stopping. Default is 5.

    Methods:
        get_motifs(data: Dataset, path: str, name: str) -> pd.DataFrame:
            Fetches motifs from the given dataset and saves them to a CSV file.
            Returns a pandas DataFrame containing the fetched motifs.

    Attributes:
        SEARCH_MOTIF_URL (str): The URL of the motif search endpoint.

    Idea from DeepDrugDomain - https://github.com/yazdanimehdi/DeepDrugDomain
    """

    SEARCH_MOTIF_URL = "https://www.genome.jp/tools-bin/search_motif_lib"

    def __init__(self, concurrent_sessions=2, max_unsuccessful_responses=5):
        self.concurrent_sessions = concurrent_sessions
        self.max_unsuccessful_responses = max_unsuccessful_responses
        self.unsuccessful_responses = 0
        self.stop_fetching = False

    async def _get_motifs_async(
        self, sequence, session, semaphore, target_id, motif_file_path
    ):
        # Form data for the POST request
        form_data = {
            "seq": sequence,
            "FORMAT": "PROSITE",
            "prosite_pattern": "on",
            "pfam": "on",
            "skip_entry": "on",
            "skip_unspecific_profile": "on",
        }

        async with semaphore:
            if self.stop_fetching:
                return

            # Send POST request to fetch motifs
            async with session.post(self.SEARCH_MOTIF_URL, data=form_data) as response:
                if not response.ok:
                    self.unsuccessful_responses += 1
                    if self.unsuccessful_responses >= self.max_unsuccessful_responses:
                        print("Max unsuccessful responses reached. Stopping.")
                        self.stop_fetching = True
                    return

                # Parse the response to extract motifs
                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                motifs = soup.find_all("input", {"type": "hidden", "name": "FOUND"})

                # Get the first motif if available, otherwise set to "NaN"
                motif = motifs[0].get("value").split(",")[0] if motifs else None
                if motif:
                    await self._write_to_file(target_id, motif, motif_file_path)
                else:
                    await self._write_to_file(
                        target_id, motif="NaN", motif_file_path=motif_file_path
                    )

        return

    async def _write_to_file(self, target_id, motif, motif_file_path):
        # Write the target ID and motif to the file
        async with aiofiles.open(motif_file_path, mode="a") as file:
            content = f'"{target_id}","{motif}"\n'
            await file.write(content)

    async def _fetch_motifs_async(
        self, sequences, target_ids, motif_file_path, progress_bar=None
    ):
        semaphore = asyncio.Semaphore(self.concurrent_sessions)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for seq, target_id in zip(sequences, target_ids):
                # Create a task for each sequence to fetch motifs asynchronously
                task = asyncio.create_task(
                    self._get_motifs_async(
                        seq, session, semaphore, target_id, motif_file_path
                    )
                )
                # Update the progress bar if provided
                task.add_done_callback(
                    lambda x: progress_bar.update(1) if progress_bar else None
                )
                tasks.append(task)
                await asyncio.sleep(0.1)  # Small delay to avoid overwhelming the server
            await asyncio.gather(*tasks)
            if self.stop_fetching:
                print("Stopped due to too many unsuccessful responses.")

        return

    def _update_motif_file(self, data, motif_file_path, local_data=pd.DataFrame | None):
        initial = 0
        if isinstance(local_data, pd.DataFrame) and not local_data.empty:
            # Filter out already processed data
            unprocessed_data = data[
                ~data["Target_ID"].isin(local_data["Target_ID"])
            ].dropna()
            initial += len(local_data["Target_ID"])
        else:
            # If no local data, initialize the motif file with headers
            unprocessed_data = data
            with open(motif_file_path, "w") as file:
                file.write("Target_ID,Motif\n")

        # Extract sequences and target IDs from the unprocessed data
        sequences = unprocessed_data["Target"]
        target_ids = unprocessed_data["Target_ID"]

        if target_ids.empty:
            # If there are no new target IDs to process, indicate that the file is up to date
            print("Motif file up to date.")
            return True
        else:
            # Initialize a progress bar for fetching motifs
            progress_bar = tqdm(
                total=len(data["Target_ID"]),
                initial=initial,
                desc="Fetching motifs",
            )

            # Run the asynchronous motif fetching process
            asyncio.run(
                self._fetch_motifs_async(
                    sequences, target_ids, motif_file_path, progress_bar
                )
            )

        return False

    def _filter_data(self, data):
        # Remove rows with NaN values in the 'Target' column
        filtered_data = data.dropna(subset=["Target"])
        # Remove duplicate rows based on 'Target_ID'
        filtered_data.drop_duplicates(subset=["Target_ID"], inplace=True)

        return filtered_data

    def get_motifs(self, data: Dataset, path, name):
        # Filter the data to remove NaN values and duplicates
        filtered_data = self._filter_data(data)

        # Define the path for the motif file
        motif_file_path = Path(path, f"{name.lower()}_motifs.csv")

        # Check if the motif file already exists and is not empty
        if motif_file_path.is_file() and motif_file_path.stat().st_size > 0:
            # Load existing motif data from the file
            local_data = pd.read_csv(motif_file_path)
            print("Motif file loaded successfully.")

            # Update the motif file with any new data
            updated = self._update_motif_file(
                filtered_data, motif_file_path=motif_file_path, local_data=local_data
            )
        else:
            # Create the directory if it doesn't exist
            motif_file_path.parent.mkdir(exist_ok=True, parents=True)

            # Create a new motif file and populate it with data
            updated = self._update_motif_file(filtered_data, motif_file_path)

        # Reload the motif data if it was updated
        if updated:
            local_data = pd.read_csv(motif_file_path)

        # Return the motif data as a pandas DataFrame
        return local_data
