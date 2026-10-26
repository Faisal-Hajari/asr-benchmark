from asr_benchmark.datasets import DACS, ESCWA, MASC, MGB2, MGB3, SADA, Casablanca, CommonVoice, Mixat
from asr_benchmark.evaluation import evaluate, normalize_arabic_text, normalize_spelling_text
from asr_benchmark.transcriber import Transcriber, LiteLLMTranscriber
from asr_benchmark.transcription_client import QWEN3_ASR_CHAT_TEMPLATE, VLLMChatCompletionsClient, VLLMTranscriptionsClient
from asr_benchmark.transforms import (
    Compose, Resample, ToMono, qwen_asr_postprocessing, qwen_asr_preprocessing, voxtral_postprocessing
)
import json
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

# Each dataset is built only when the loop reaches it, so a failure in one does not cost loading the others.
BENCHMARKS = {
    "Leaderboard": {
        "SADA": SADA,
        "Casablanca": Casablanca,
        "MASC clean": lambda: MASC("clean"),
        "MASC noisy": lambda: MASC("noisy"),
        "MGB2": MGB2,
        "CommonVoice17": CommonVoice,
    },
    "Arabic ASR": {
        "MGB3": MGB3,
        "DACS": DACS,
        # All 1,585 test clips: 768 Arabic only and 817 code-switched.
        "Mixat": Mixat,
    },
    "Code Switch AR-EN": {
        "ESCWA": ESCWA,
        "Mixat (CS)": lambda: Mixat(language="CS"),
    },
}
# Only the leaderboard's six test sets are averaged, matching how the leaderboard reports its average.
AVERAGED = {"Leaderboard"}
# The leaderboard keeps its own normalization so its numbers stay comparable with published ones.
NORMALIZERS = {
    "Leaderboard": normalize_arabic_text,
    "Arabic ASR": normalize_spelling_text,
    "Code Switch AR-EN": normalize_spelling_text,
}
# (row label, scale, format) for each value evaluate returns, in order.
METRICS = [("WER %", 100, ".2f"), ("CER %", 100, ".2f"), ("RTF", 1, ".4f"), ("Latency s", 1, ".3f"), ("Audio min/s", 1, ".2f")]
NUM_WORKERS = 16
BATCH_SIZE = 16
LANG= "ar"
MAX_NUM_TOKEN=256

def main(
    transcriber: Transcriber,
    cache_path: str | None = None,
) -> str:
    """Evaluates every benchmark, prints its tables and returns them as markdown.

    With cache_path, each dataset's metrics are saved there as soon as it finishes and datasets
    already saved are skipped, so a run that fails part way resumes where it stopped.
    """
    cache = json.loads(Path(cache_path).read_text()) if cache_path and Path(cache_path).exists() else {}
    results = {}
    for benchmark, datasets in BENCHMARKS.items():
        results[benchmark] = {}
        for name, make_dataset in datasets.items():
            key = f"{benchmark}/{name}"
            if key not in cache:
                cache[key] = evaluate(transcriber, make_dataset(), batch_size=BATCH_SIZE, normalize=NORMALIZERS[benchmark])
                if cache_path:
                    Path(cache_path).write_text(json.dumps(cache, indent=1))
            results[benchmark][name] = tuple(cache[key])
    tables = []
    for benchmark, scores in results.items():
        if benchmark in AVERAGED:
            scores = {"Avg": tuple(sum(column) / len(column) for column in zip(*scores.values())), **scores}
        tables.append(format_table(benchmark, scores))
    report = "\n\n".join(tables)
    print(report)
    return report


def format_table(title: str, scores: dict[str, tuple[float, float, float, float, float]]) -> str:
    """Formats a markdown table with one column per dataset and one row per metric."""
    names = ["Metric", *scores]
    widths = [max(len(name), 11) for name in names]
    rows = [names, ["-" * width for width in widths]]
    for index, (label, scale, fmt) in enumerate(METRICS):
        rows.append([label, *(format(values[index] * scale, fmt) for values in scores.values())])
    lines = ["| " + " | ".join(cell.rjust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
    return f"### {title}\n\n" + "\n".join(lines)


@dataclass(frozen=True)
class ModelSetup:
    """A model to benchmark: its id, the extra vllm.sh flags it is served with (None for a model that
    runs in this process), and its transcriber."""

    model: str
    serve_args: tuple[str, ...] | None
    make_transcriber: Callable[[str], Transcriber]


def transcription_endpoint(model: str) -> Transcriber:
    # NB: 16 kHz mono on the client keeps long 44.1 kHz stereo clips (Mixat has a 260 s one, 43.7 MB)
    # under vLLM's 25 MB upload limit; the server converts to 16 kHz mono itself anyway.
    client = VLLMTranscriptionsClient(model, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN)
    return LiteLLMTranscriber(client, language=LANG, transform=Compose([ToMono(), Resample(16000)]))


def voxtral_realtime(model: str) -> Transcriber:
    # NB: imported here so runs of vLLM-served models do not load torch.
    from asr_benchmark.hf_transcriber import VoxtralRealtimeTranscriber

    return VoxtralRealtimeTranscriber(model, text_transform=voxtral_postprocessing())


def qwen3_asr(model: str) -> Transcriber:
    """Qwen3-ASR and its fine-tunes (Audar), reproducing the qwen-asr reference setup."""
    client = VLLMChatCompletionsClient(
        model,
        max_workers=NUM_WORKERS,
        max_num_tokens=MAX_NUM_TOKEN,
        chat_template=QWEN3_ASR_CHAT_TEMPLATE,
        language_names={"ar": "Arabic"},
        wav_subtype="FLOAT",
    )
    return LiteLLMTranscriber(
        client, language=LANG, transform=qwen_asr_preprocessing(), text_transform=qwen_asr_postprocessing()
    )


QWEN3_ASR_SERVE_ARGS = (
    "--trust-request-chat-template", "--max-model-len", "8192", "--max-num-batched-tokens", "8192"
)
MODELS = {
    "Whisper large-v3": ModelSetup("openai/whisper-large-v3", (), transcription_endpoint),
    "Cohere Transcribe Arabic": ModelSetup("CohereLabs/cohere-transcribe-arabic-07-2026", (), transcription_endpoint),
    # NB: the client sends QWEN3_ASR_CHAT_TEMPLATE with each request, which vLLM refuses without this flag.
    # NB: max-model-len 8192 as on the Audar card; the model's own maximum needs more KV cache than 12 GB
    # leaves. max-num-batched-tokens 8192 also sizes the audio encoder cache, which at the default 2048
    # rejects clips over ~160 s; qwen-asr sends clips up to 1200 s whole, and Mixat has a 260 s one.
    # NB: Audar's 4.4 GB of weights leave no KV cache at 8192 on a 12 GB card; 4096 still fits the 260 s clip.
    "Audar-ASR-V1-Turbo": ModelSetup(
        "audarai/Audar-ASR-V1-Turbo",
        ("--trust-request-chat-template", "--max-model-len", "8192", "--max-num-batched-tokens", "4096"),
        qwen3_asr,
    ),
    "Qwen3-ASR 0.6B": ModelSetup("Qwen/Qwen3-ASR-0.6B", QWEN3_ASR_SERVE_ARGS, qwen3_asr),
    # In this process with transformers, not vLLM: vLLM 0.30 mixes text between concurrent requests
    # for this model, and one request at a time runs at RTF 0.64 on a 12 GB card.
    "Voxtral Mini 4B Realtime Arabic": ModelSetup("mistralai/Voxtral-Mini-4B-Realtime-Arabic", None, voxtral_realtime),
    # Voxtral Mini with the model card's recommended setup (transcription endpoint, temperature 0).
    "Voxtral Mini 3B": ModelSetup(
        "mistralai/Voxtral-Mini-3B-2507",
        ("--tokenizer_mode", "mistral", "--config_format", "mistral", "--load_format", "mistral",
         "--max-model-len", "2048", "--max-num-seqs", "16", "--enforce-eager"),
        transcription_endpoint,
    ),
}
MODEL = "Qwen3-ASR 0.6B"


if __name__ == "__main__":
    # Serve first with: MODEL=<setup.model> bash vllm.sh <setup.serve_args>
    setup = MODELS[MODEL]
    main(setup.make_transcriber(setup.model))
