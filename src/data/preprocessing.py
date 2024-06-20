import asyncio
import aiohttp
from bs4 import BeautifulSoup
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import threading


class MotifFetcher:
    SEARCH_MOTIF_URL = "https://www.genome.jp/tools-bin/search_motif_lib"

    def __init__(self, concurrent_sessions=2, max_unsuccessful_responses=5):
        self.concurrent_sessions = concurrent_sessions
        self.max_unsuccessful_responses = max_unsuccessful_responses
        self.unsuccessful_responses = 0
        self.stop_fetching = False
        self.file_lock = threading.Lock()

    async def _get_motifs_async(
        self, sequence, session, semaphore, target_id, motif_file_path
    ):
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

            async with session.post(self.SEARCH_MOTIF_URL, data=form_data) as response:
                print("here")
                if not response.ok:
                    self.unsuccessful_responses += 1
                    if self.unsuccessful_responses >= self.max_unsuccessful_responses:
                        print("Max unsuccessful responses reached. Stopping.")
                        self.stop_fetching = True
                    return

                text = await response.text()
                soup = BeautifulSoup(text, "html.parser")
                motifs = soup.find_all("input", {"type": "hidden", "name": "FOUND"})

                motif = motifs[0].get("value").split(",")[0] if motifs else None
                if motif:
                    self._write_motif_to_file(target_id, motif, motif_file_path)
                else:
                    self._write_motif_to_file(
                        target_id, motif="NaN", motif_file_path=motif_file_path
                    )

    def _write_motif_to_file(self, target_id, motif, motif_file_path):
        with self.file_lock:
            with open(motif_file_path, "a") as file:
                file.write(f"{target_id},{motif}\n")

    async def _fetch_motifs_async(
        self, sequences, target_ids, motif_file_path, progress_bar=None
    ):
        semaphore = asyncio.Semaphore(self.concurrent_sessions)
        async with aiohttp.ClientSession() as session:
            tasks = []
            for seq, target_id in zip(sequences, target_ids):
                task = asyncio.create_task(
                    self._get_motifs_async(
                        seq, session, semaphore, target_id, motif_file_path
                    )
                )
                task.add_done_callback(
                    lambda x: progress_bar.update(1) if progress_bar else None
                )
                tasks.append(task)
                await asyncio.sleep(0.1)
            await asyncio.gather(*tasks)
            if self.stop_fetching:
                print("Stopped due to too many unsuccessful responses.")

    def _update_motif_file(
        self, dataset, motif_file_path, local_data=pd.DataFrame | None
    ):
        initial = 0
        if isinstance(local_data, pd.DataFrame) and not local_data.empty:
            unprocessed_data = dataset.data[
                ~dataset.data["Target_ID"].isin(local_data["Target_ID"])
            ]
            initial += len(local_data["Target_ID"])
        else:
            unprocessed_data = dataset.data
            with open(motif_file_path, "a") as file:
                file.write("Target_ID,Motif\n")

        sequences = unprocessed_data["Target"]
        target_ids = unprocessed_data["Target_ID"]

        progress_bar = tqdm(
            total=len(dataset.data["Target_ID"]),
            initial=initial,
            desc="Fetching motifs",
        )

        "there"
        asyncio.run(
            self._fetch_motifs_async(
                sequences, target_ids, motif_file_path, progress_bar
            )
        )

    def load_motif_file(self, dataset):
        motif_file_path = Path(dataset.path, f"{dataset.name.lower()}_motifs.csv")

        if motif_file_path.is_file() and motif_file_path.stat().st_size > 0:
            local_data = pd.read_csv(motif_file_path)
            print("Motif file loaded successfully.")
            df = self._update_motif_file(
                dataset, motif_file_path=motif_file_path, local_data=local_data
            )
        else:
            df = self._update_motif_file(dataset, motif_file_path)

        return df
