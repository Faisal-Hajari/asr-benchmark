import csv
import dataclasses
import io
import re
import tarfile
from pathlib import Path

import datasets
from huggingface_hub import hf_hub_download

from asr_benchmark.audio import Audio, load


class ASRDataset:
    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> Audio:
        raise NotImplementedError


class SADA(ASRDataset):
    """SADA 2022 Saudi TV speech from the Hugging Face mirror.

    Args:
        split: "train", "validation" or "test".
        repo_id: Hugging Face dataset repository.
        text_column: "cleaned_text" matches the leaderboard's references, "text" keeps punctuation.
    """

    def __init__(
        self,
        split: str = "test",
        repo_id: str = "MohamedRashad/SADA22",
        text_column: str = "cleaned_text",
    ):
        # NB: naming the split's files keeps load_dataset from downloading every split (train is 27 GB);
        # no_checks stops it from failing because the card also lists the splits left out.
        dataset = datasets.load_dataset(
            repo_id,
            data_files={split: f"data/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
        # NB: decode=False yields the raw WAV bytes; decoding in datasets would require torchcodec and torch.
        dataset = dataset.select_columns(["audio", text_column]).cast_column("audio", datasets.Audio(decode=False))
        # NB: rows without a transcript store the string "nan"; the leaderboard drops them too.
        self.dataset = dataset.filter(lambda text: text != "nan", input_columns=text_column)
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Audio:
        row = self.dataset[index]
        audio = load(io.BytesIO(row["audio"]["bytes"]))
        return dataclasses.replace(audio, transcription=row[self.text_column])


class Casablanca(ASRDataset):
    """Casablanca multidialectal speech (8 countries) from the official Hugging Face repository.

    Args:
        split: "validation" or "test".
        repo_id: Hugging Face dataset repository.
        excluded_pattern: Regex; clips whose transcription matches it are dropped. The default, a
            one-word English tag such as [music], drops exactly the 92 test clips missing from the
            leaderboard's casablanca_test.json. Arabic tags ([موسيقى]) and bracketed French stay.
    """

    def __init__(
        self,
        split: str = "test",
        repo_id: str = "UBC-NLP/Casablanca",
        excluded_pattern: str = r"\[[A-Za-z]+\]",
    ):
        # NB: naming the split's files keeps load_dataset from downloading the other split.
        dataset = datasets.load_dataset(
            repo_id,
            data_files={split: f"*/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
        dataset = dataset.select_columns(["audio", "transcription"]).cast_column("audio", datasets.Audio(decode=False))
        excluded = re.compile(excluded_pattern)
        self.dataset = dataset.filter(lambda text: excluded.search(text) is None, input_columns="transcription")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Audio:
        row = self.dataset[index]
        audio = load(io.BytesIO(row["audio"]["bytes"]))
        return dataclasses.replace(audio, transcription=row["transcription"])


class MASC(ASRDataset):
    """MASC YouTube speech from the Hugging Face mirror, clean or noisy test subset.

    The mirror is a loading script, which datasets>=4 no longer runs, so the
    transcript CSV and the audio archive are fetched directly; the archive is
    extracted once into cache_dir.

    Args:
        subset: "clean" or "noisy", matching masc_test_clean.json and masc_test_noisy.json.
        split: "test" or "dev".
        repo_id: Hugging Face dataset repository.
        cache_dir: Where the audio archive is extracted.
    """

    def __init__(
        self,
        subset: str = "clean",
        split: str = "test",
        repo_id: str = "pain/MASC",
        cache_dir: str = "data/masc",
    ):
        transcript = hf_hub_download(repo_id, f"transcript/{split}/{split}.csv", repo_type="dataset")
        with open(transcript, encoding="utf-8") as f:
            rows = list(csv.DictReader(f, quoting=csv.QUOTE_NONE))
        subset_type = {"clean": "c", "noisy": "n"}[subset]
        self.rows = [row for row in rows if row["type"] == subset_type]

        # NB: check every file, not the directory, so an interrupted extraction is redone.
        self.audio_dir = Path(cache_dir) / split / "audios"
        if not all((self.audio_dir / row["file_path"]).exists() for row in rows):
            archive = hf_hub_download(repo_id, f"audio/{split}/{split}_01.tar.gz", repo_type="dataset")
            with tarfile.open(archive) as tar:
                tar.extractall(self.audio_dir.parent, filter="data")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Audio:
        row = self.rows[index]
        audio = load(str(self.audio_dir / row["file_path"]))
        return dataclasses.replace(audio, transcription=row["text"])
