"""Evaluation replicating the Open Universal Arabic ASR Leaderboard.

Reference: https://github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard
(eval.py and models/*.py at commit 8fce859).
"""

import re
import time

import editdistance
from tqdm import tqdm

from asr_benchmark.datasets import ASRDataset
from asr_benchmark.transcriber import Transcriber

EASTERN_TO_WESTERN_NUMERALS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_arabic_text(text: str) -> str:
    # NB: order and regexes match the reference eval.py exactly; changing either
    # makes results incomparable with the leaderboard.
    text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~،؛؟]', "", text)
    text = re.sub(r"[ً-ْ]", "", text)  # diacritics
    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)
    text = text.translate(EASTERN_TO_WESTERN_NUMERALS)
    text = re.sub(r"(^|\s)و\s+", r"\1و", text)  # attach the conjunction و to the next word
    return re.sub(r"\s+", " ", text).strip()


def word_error_rate(hypotheses: list[str], references: list[str], use_cer: bool = False) -> float:
    """Corpus-level error rate: total edits over total reference tokens.

    Port of nemo.collections.asr.metrics.wer.word_error_rate. With use_cer the
    tokens are characters, spaces included.
    """
    if len(hypotheses) != len(references):
        raise ValueError(f"got {len(hypotheses)} hypotheses and {len(references)} references")
    edits = 0
    tokens = 0
    for hypothesis, reference in zip(hypotheses, references):
        hypothesis_tokens = list(hypothesis) if use_cer else hypothesis.split()
        reference_tokens = list(reference) if use_cer else reference.split()
        tokens += len(reference_tokens)
        edits += editdistance.eval(reference_tokens, hypothesis_tokens)
    return edits / tokens if tokens else float("inf")


def evaluate(
    transcriber: Transcriber,
    dataset: ASRDataset,
    batch_size: int = 16,
    warmup: int = 5,
) -> tuple[float, float, float, float, float]:
    """Transcribes every clip in the dataset and scores it against its transcription.

    WER and CER cover every clip, with both sides normalized as in the reference
    eval.py. Timing metrics leave out batches starting within the first `warmup`
    clips. batch_size=1 reproduces the reference's one clip per call, which is
    the only setting where RTF is comparable with the leaderboard.

    Returns:
        (WER, CER, RTF, average latency in seconds, throughput in audio minutes per second).
        RTF is summed request latency over summed audio duration; throughput is
        audio minutes over wall-clock seconds spent in transcribe calls.
    """
    references = []
    predictions = []
    latency_s = []
    audio_s = 0.0
    wall_s = 0.0
    with tqdm(total=len(dataset), desc=type(dataset).__name__) as progress:
        for start_index in range(0, len(dataset), batch_size):
            batch = [dataset[i] for i in range(start_index, min(start_index + batch_size, len(dataset)))]

            start = time.perf_counter()
            transcriptions = transcriber.transcribe(batch)
            elapsed = time.perf_counter() - start

            if start_index >= warmup:
                wall_s += elapsed
                audio_s += sum(audio.audio_length_s for audio in batch)
                latency_s += [t.inference_time_s for t in transcriptions]
            references += [normalize_arabic_text(t.audio.transcription) for t in transcriptions]
            predictions += [normalize_arabic_text(t.text) for t in transcriptions]
            progress.update(len(batch))

    return (
        word_error_rate(predictions, references),
        word_error_rate(predictions, references, use_cer=True),
        sum(latency_s) / audio_s,
        sum(latency_s) / len(latency_s),
        audio_s / 60 / wall_s,
    )
