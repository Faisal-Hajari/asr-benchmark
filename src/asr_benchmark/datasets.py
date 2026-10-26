import csv
import dataclasses
import io
import re
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import soundfile as sf
from huggingface_hub import hf_hub_download

from asr_benchmark.audio import Audio, load


class ASRDataset:
    # True when the dataset is built around code-switching (e.g. Arabic-English within a sentence).
    code_switching: bool = False

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


class CommonVoice(ASRDataset):
    """Mozilla Common Voice from a Hugging Face mirror, one language and split.

    The leaderboard uses Common Voice 18.0; this 17.0 mirror's Arabic test split shares 10,410 of
    its 10,471 clips, with labels equal after normalization once removed_characters are stripped.

    Args:
        language: Common Voice locale, e.g. "ar".
        split: "train", "validation" or "test".
        repo_id: Hugging Face dataset repository.
        removed_characters: Stripped from every sentence. «» survives the leaderboard's
            punctuation regex, and its own manifest has them removed.
    """

    def __init__(
        self,
        language: str = "ar",
        split: str = "test",
        repo_id: str = "fixie-ai/common_voice_17_0",
        removed_characters: str = "«»",
    ):
        # NB: naming the split's files keeps load_dataset from downloading other languages and splits.
        dataset = datasets.load_dataset(
            repo_id,
            data_files={split: f"{language}/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
        self.dataset = dataset.select_columns(["audio", "sentence"]).cast_column("audio", datasets.Audio(decode=False))
        self.removed_characters = str.maketrans("", "", removed_characters)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Audio:
        row = self.dataset[index]
        # NB: the audio is MP3 at 32 kHz; soundfile decodes it with libsndfile >= 1.1.
        audio = load(io.BytesIO(row["audio"]["bytes"]))
        return dataclasses.replace(audio, transcription=row["sentence"].translate(self.removed_characters))


class MGB2(ASRDataset):
    """MGB-2 Al Jazeera broadcast test set from QCRI's gated Hugging Face repository.

    The archive holds 17 full programs with Kaldi-style segments and text files. It is
    extracted once into cache_dir, and each segment is read straight out of its program.
    The dev archive is laid out differently and is not supported.

    Args:
        segmentation: "non_overlap_speech" is the official MGB-2 scoring set and matches the
            leaderboard's mgb2_test.json exactly; "all" adds the segments with overlapping speakers.
        repo_id: Hugging Face dataset repository; access must be requested on its page first.
        cache_dir: Where the archive is extracted.
    """

    def __init__(
        self,
        segmentation: str = "non_overlap_speech",
        repo_id: str = "QCRI/mgb2",
        cache_dir: str = "data/mgb2",
    ):
        self.root = Path(cache_dir) / "test"
        # NB: check every program, not the directory, so an interrupted extraction is redone.
        wav_list = self.root / "wav.scp"
        if not wav_list.exists() or not all((self.root / line.split()[1]).exists() for line in open(wav_list)):
            archive = hf_hub_download(repo_id, "test/test.tar.gz", repo_type="dataset")
            with tarfile.open(archive) as tar:
                tar.extractall(cache_dir, filter="data")

        self.programs = {program: self.root / path for program, path in (line.split() for line in open(wav_list))}
        self.sample_rates = {program: sf.info(path).samplerate for program, path in self.programs.items()}
        with open(self.root / f"text.{segmentation}", encoding="utf-8") as f:
            texts = dict(line.rstrip("\n").split(" ", 1) for line in f)
        with open(self.root / f"segments.{segmentation}", encoding="utf-8") as f:
            self.segments = [
                (segment, program, float(start_s), float(end_s), texts[segment])
                for segment, program, start_s, end_s in (line.split() for line in f)
            ]

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Audio:
        _, program, start_s, end_s, text = self.segments[index]
        sample_rate = self.sample_rates[program]
        audio = load(str(self.programs[program]), start=round(start_s * sample_rate), stop=round(end_s * sample_rate))
        return dataclasses.replace(audio, transcription=text)


class MGB3(ASRDataset):
    """MGB-3 Egyptian Arabic YouTube speech (7 genres) from the Hugging Face mirror; not on the leaderboard.

    Args:
        split: "train", "validation" or "test".
        repo_id: Hugging Face dataset repository.
        removed_pattern: Regex removed from every transcription. The default drops words marked
            as cut off with # (أ#, ال#), which the leaderboard normalizer would otherwise turn into
            words a model is not expected to say. "" keeps the transcriptions as they are.
    """

    def __init__(
        self,
        split: str = "test",
        repo_id: str = "MohamedRashad/MGB-3-Arabic",
        removed_pattern: str = r"\S*#\S*",
    ):
        # NB: naming the split's files keeps load_dataset from downloading the other splits.
        dataset = datasets.load_dataset(
            repo_id,
            data_files={split: f"data/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
        self.dataset = dataset.select_columns(["audio", "text"]).cast_column("audio", datasets.Audio(decode=False))
        self.removed = re.compile(removed_pattern)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Audio:
        row = self.dataset[index]
        audio = load(io.BytesIO(row["audio"]["bytes"]))
        text = " ".join(self.removed.sub("", row["text"]).split())
        return dataclasses.replace(audio, transcription=text)


class ESCWA(ASRDataset):
    """ESCWA-CS: 2.8 hours of UN ESCWA meetings (2019) from QCRI's gated Hugging Face repository; not on the leaderboard.

    Dialectal Arabic with intra-sentential English, and French from Maghrebi speakers. The
    archive holds 25 recordings with Kaldi-style segments and text files; it is extracted once
    into cache_dir and each segment is read straight out of its recording.

    Args:
        repo_id: Hugging Face dataset repository; access must be requested on its page first.
        cache_dir: Where the archive is extracted.
        removed_pattern: Regex removed from every transcription. The default drops the <UNK>
            marker for unintelligible speech, which the leaderboard normalizer would turn into
            the word UNK. "" keeps the transcriptions as they are.
    """

    code_switching = True

    def __init__(
        self,
        repo_id: str = "QCRI/escwa",
        cache_dir: str = "data/escwa",
        removed_pattern: str = r"<UNK>",
    ):
        self.root = Path(cache_dir) / "cs.released"
        segments_file = self.root / "segments"
        # NB: check every recording, not the directory, so an interrupted extraction is redone.
        if not segments_file.exists() or not all(
            (self.root / "wav" / f"{line.split()[1]}.wav").exists() for line in open(segments_file)
        ):
            archive = hf_hub_download(repo_id, "cs_escwa.tar.gz", repo_type="dataset")
            with tarfile.open(archive) as tar:
                tar.extractall(cache_dir, filter="data")

        removed = re.compile(removed_pattern)
        with open(self.root / "text", encoding="utf-8") as f:
            texts = {segment: " ".join(removed.sub("", text).split()) for segment, _, text in (line.partition(" ") for line in f)}
        # NB: text has one transcription more than segments; it has no audio and is left out.
        with open(segments_file, encoding="utf-8") as f:
            self.segments = [
                (recording, float(start_s), float(end_s), texts[segment])
                for segment, recording, start_s, end_s in (line.split() for line in f)
            ]
        self.sample_rates = {
            recording: sf.info(self.root / "wav" / f"{recording}.wav").samplerate for recording, *_ in self.segments
        }

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Audio:
        recording, start_s, end_s, text = self.segments[index]
        sample_rate = self.sample_rates[recording]
        path = str(self.root / "wav" / f"{recording}.wav")
        audio = load(path, start=round(start_s * sample_rate), stop=round(end_s * sample_rate))
        return dataclasses.replace(audio, transcription=text)



class Mixat(ASRDataset):
    """Mixat: Emirati Arabic podcasts code-mixed with English, from the authors' Hugging Face release.

    Each clip has the transcript three ways: English in Latin inside brackets ("transcript",
    the original), English transliterated into Arabic script ("transliteration"), and fully
    translated into Arabic ("translation"). Clips are labelled "CS" or "Arabic".

    Args:
        split: "train" or "test".
        repo_id: Hugging Face dataset repository.
        text_column: "transcript", "transliteration" or "translation".
        language: "CS" or "Arabic" keeps only clips with that label; None keeps all.
    """

    code_switching = True

    def __init__(
        self,
        split: str = "test",
        repo_id: str = "sqrk/mixat-tri",
        text_column: str = "transcript",
        language: str | None = None,
    ):
        # NB: naming the split's files keeps load_dataset from downloading the other split.
        dataset = datasets.load_dataset(
            repo_id,
            data_files={split: f"data/{split}-*.parquet"},
            split=split,
            verification_mode="no_checks",
        )
        dataset = dataset.select_columns(["audio", text_column, "language"]).cast_column(
            "audio", datasets.Audio(decode=False)
        )
        if language is not None:
            dataset = dataset.filter(lambda label: label == language, input_columns="language")
        self.dataset = dataset
        self.text_column = text_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Audio:
        row = self.dataset[index]
        # NB: the audio is 44.1 kHz stereo; servers downmix and resample it themselves.
        audio = load(io.BytesIO(row["audio"]["bytes"]))
        return dataclasses.replace(audio, transcription=row[self.text_column])

# Buckwalter transliteration to Arabic script, for the letters DACS uses plus the common extensions.
BUCKWALTER_TO_ARABIC = str.maketrans(
    "'|>&<}AbptvjHxd*rzs$SDTZEg_fqklmnhwYyFNKauio~`{PJVG",
    "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىيًٌٍَُِّْٰٱپچڤگ",
)


class DACS(ASRDataset):
    """DACS: 2 hours of Egyptian broadcast speech with word-level code-switching labels; not on the leaderboard.

    Manually segmented and transcribed utterances of the MGB-3 ADI-5 dev split (EGY), from
    github.com/qcri/Arabic_speech_code_switching. Speakers switch between MSA and Egyptian
    Arabic, with a few foreign words. Metadata comes from the repository and the 315 source
    clips from alt.qcri.org; both are downloaded once into cache_dir.

    Args:
        cache_dir: Where the metadata and audio are stored.
        repo_url: Raw-file base of the repository, pinned to a commit.
        max_workers: Concurrent audio downloads.
    """

    code_switching = True

    def __init__(
        self,
        cache_dir: str = "data/dacs",
        repo_url: str = "https://raw.githubusercontent.com/qcri/Arabic_speech_code_switching/26548bc",
        max_workers: int = 16,
    ):
        self.root = Path(cache_dir)
        metadata = ["segments_dacs_selected", "DACS_word_level.feat", "mgb3_audio_list.txt"]
        with ThreadPoolExecutor(max_workers) as pool:
            list(pool.map(lambda name: download(f"{repo_url}/{name}", self.root / name), metadata))
            urls = (self.root / "mgb3_audio_list.txt").read_text().split()
            list(pool.map(lambda url: download(url, self.root / "audio" / url.rsplit("/", 1)[1]), urls))

        # NB: the feat file has one line per phone, so each word repeats; word_index orders them.
        words = {}
        with open(self.root / "DACS_word_level.feat", encoding="utf-8") as f:
            for line in f:
                if not line.startswith("#"):
                    utterance, word_index, _, word = line.split()[:4]
                    words.setdefault(utterance, {})[int(word_index)] = word.translate(BUCKWALTER_TO_ARABIC)
        with open(self.root / "segments_dacs_selected", encoding="utf-8") as f:
            self.segments = [
                (clip, float(start_s), float(end_s), " ".join(w for _, w in sorted(words[utterance].items())))
                for utterance, clip, start_s, end_s in (line.split() for line in f)
            ]

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> Audio:
        clip, start_s, end_s, text = self.segments[index]
        path = str(self.root / "audio" / f"{clip}.wav")
        sample_rate = sf.info(path).samplerate
        audio = load(path, start=round(start_s * sample_rate), stop=round(end_s * sample_rate))
        return dataclasses.replace(audio, transcription=text)


def download(url: str, path: Path) -> None:
    """Downloads url to path unless it exists; a partial download never takes the final name."""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    urllib.request.urlretrieve(url, partial)
    partial.replace(path)
