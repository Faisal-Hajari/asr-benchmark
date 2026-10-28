from asr_benchmark.datasets import DACS, ESCWA, MASC, MGB2, MGB3, SADA, ASRDataset, Casablanca, CommonVoice, Mixat
from asr_benchmark.evaluation import evaluate, normalize_arabic_text, normalize_spelling_text
from asr_benchmark.transcriber import Transcriber, LiteLLMTranscriber
from asr_benchmark.transcription_client import QWEN3_ASR_CHAT_TEMPLATE, VLLMChatCompletionsClient, VLLMTranscriptionsClient
from asr_benchmark.transforms import (
    Compose, DropLanguageTag, Resample, ToMono, qwen_asr_postprocessing, qwen_asr_preprocessing, voxtral_postprocessing
)
import json
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import constants

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

def load_datasets(
    max_workers: int = 4, after_wave: Callable[[], None] | None = None, keys: set[str] | None = None
) -> dict[str, ASRDataset | Exception]:
    """Builds every dataset in BENCHMARKS, or only those whose "benchmark/name" is in keys, downloading
    max_workers at a time, keyed "benchmark/name".

    after_wave runs after each max_workers downloads, e.g. drop_hub_downloads to free their raw files.
    A dataset that fails to build, e.g. a gated repository without access, maps to its exception.
    """
    # NB: datasets named alike share a download (MASC clean/noisy, Mixat/Mixat (CS)), so they build in
    # one task: two threads extracting one archive into one directory would race.
    tasks: dict[str, list[tuple[str, Callable[[], ASRDataset]]]] = {}
    for benchmark, datasets in BENCHMARKS.items():
        for name, make_dataset in datasets.items():
            if keys is None or f"{benchmark}/{name}" in keys:
                tasks.setdefault(name.split()[0], []).append((f"{benchmark}/{name}", make_dataset))

    def build(task: list[tuple[str, Callable[[], ASRDataset]]]) -> dict[str, ASRDataset | Exception]:
        built = {}
        for key, make_dataset in task:
            try:
                built[key] = make_dataset()
                print(f"built {key}: {len(built[key])} clips", flush=True)
            except Exception as error:  # noqa: BLE001 - one dataset failing must not cost the others
                built[key] = error
                print(f"failed to build {key}: {error!r}", flush=True)
        return built

    waves = list(tasks.values())
    built = {}
    with ThreadPoolExecutor(max_workers) as pool:
        for start in range(0, len(waves), max_workers):
            for result in pool.map(build, waves[start:start + max_workers]):
                built.update(result)
            if after_wave:
                after_wave()
    return built


def drop_hub_downloads() -> None:
    """Empties the Hugging Face hub cache if it is inside the working directory, i.e. a scratch HF_HOME,
    freeing a wave's raw downloads. Built datasets read their own copies (the datasets cache, data/)."""
    hub = Path(constants.HF_HUB_CACHE).resolve()
    # NB: never a cache outside the working directory, which may be the team's shared one. All of it goes,
    # not just the datasets--* dirs: the file contents live in a shared hub/blobs outside them.
    if hub.is_relative_to(Path.cwd().resolve()):
        shutil.rmtree(hub, ignore_errors=True)


def main(
    transcriber: Transcriber,
    cache_path: str | None = None,
    built_datasets: dict[str, ASRDataset | Exception] | None = None,
) -> str:
    """Evaluates every benchmark, prints its tables and returns them as markdown.

    With cache_path, each dataset's metrics are saved there as soon as it finishes and datasets
    already saved are skipped, so a run that fails part way resumes where it stopped.
    built_datasets, from load_datasets, are used instead of building each dataset here; one that
    failed to build is left out of its table, and its benchmark then has no average.
    """
    cache = json.loads(Path(cache_path).read_text()) if cache_path and Path(cache_path).exists() else {}
    results = {}
    for benchmark, datasets in BENCHMARKS.items():
        results[benchmark] = {}
        for name, make_dataset in datasets.items():
            key = f"{benchmark}/{name}"
            if key not in cache:
                dataset = built_datasets[key] if built_datasets is not None else make_dataset()
                if isinstance(dataset, Exception):
                    print(f"skipping {key}: {dataset!r}", flush=True)
                    continue
                cache[key] = evaluate(transcriber, dataset, batch_size=BATCH_SIZE, normalize=NORMALIZERS[benchmark])
                if cache_path:
                    Path(cache_path).write_text(json.dumps(cache, indent=1))
            results[benchmark][name] = tuple(cache[key])
    tables = []
    for benchmark, scores in results.items():
        missing = [name for name in BENCHMARKS[benchmark] if name not in scores]
        if benchmark in AVERAGED and not missing:
            scores = {"Avg": tuple(sum(column) / len(column) for column in zip(*scores.values())), **scores}
        if scores:
            tables.append(format_table(benchmark, scores) + (f"\n\nNot run: {', '.join(missing)}" if missing else ""))
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


def format_comparison(caches: dict[str, dict[str, list[float]]]) -> str:
    """Formats one markdown table per benchmark with a row per model and a column per dataset, each cell
    "WER % / CER %". caches maps a model name to its main() cache; a dataset it has not run shows "-".
    Every table leads with an Avg column, the plain mean over its datasets, for models that ran them all."""
    tables = []
    for benchmark, datasets in BENCHMARKS.items():
        header = ["Model", "Avg", *datasets]
        rows = [header]
        for model, cache in caches.items():
            scores = [cache.get(f"{benchmark}/{name}") for name in datasets]
            average = None if None in scores else [sum(column) / len(column) for column in zip(*scores)]
            scores = [average, *scores]
            rows.append([model, *("-" if s is None else f"{s[0] * 100:.2f} / {s[1] * 100:.2f}" for s in scores)])
        widths = [max(len(row[i]) for row in rows) for i in range(len(header))]
        rows.insert(1, ["-" * width for width in widths])
        lines = ["| " + " | ".join(cell.ljust(width) for cell, width in zip(row, widths)) + " |" for row in rows]
        tables.append(f"### {benchmark} (WER % / CER %)\n\n" + "\n".join(lines))
    return "\n\n".join(tables)


@dataclass(frozen=True)
class ModelSetup:
    """A model to benchmark: its id, the extra vllm.sh flags it is served with (None for a model that
    runs in this process or on a server that is already up), and its transcriber."""

    model: str
    serve_args: tuple[str, ...] | None
    make_transcriber: Callable[[str], Transcriber]


def transcription_endpoint(model: str, api_base: str = "http://localhost:8000/v1") -> Transcriber:
    # NB: 16 kHz mono on the client keeps long 44.1 kHz stereo clips (Mixat has a 260 s one, 43.7 MB)
    # under vLLM's 25 MB upload limit; the server converts to 16 kHz mono itself anyway.
    client = VLLMTranscriptionsClient(model, api_base=api_base, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN)
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


def qwen3_asr_own_template(model: str, api_base: str) -> Transcriber:
    """A Qwen3-ASR fine-tune on a server started without --trust-request-chat-template: the model's own
    template prefills no language, so the "language {name}<asr_text>" header is dropped from the output."""
    client = VLLMChatCompletionsClient(
        model, api_base=api_base, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN, wav_subtype="FLOAT"
    )
    return LiteLLMTranscriber(
        client,
        language=LANG,
        transform=qwen_asr_preprocessing(),
        text_transform=Compose([DropLanguageTag(), qwen_asr_postprocessing()]),
    )


QWEN3_ASR_SERVE_ARGS = (
    "--trust-request-chat-template", "--max-model-len", "8192", "--max-num-batched-tokens", "8192"
)
MODELS = {
    "Whisper large-v3": ModelSetup("openai/whisper-large-v3", (), transcription_endpoint),
    "Cohere Transcribe Arabic": ModelSetup("CohereLabs/cohere-transcribe-arabic-07-2026", (), transcription_endpoint),
    # The local STT that aiva-stt-cohere.service already serves: the same Cohere model on vLLM at :8004
    # under the name "cohere" (the HF id gets a 404 there). :8000 on this host is an LLM, not vllm.sh.
    "Cohere Transcribe Arabic (aiva-stt)": ModelSetup(
        "cohere", None, lambda model: transcription_endpoint(model, api_base="http://127.0.0.1:8004/v1")
    ),
    # The in-house Audar/Qwen3-ASR fine-tune that aiva-stt-lucidya.service already serves on :8008.
    # NB: that unit has no --trust-request-chat-template, so qwen3_asr's per-request template gets a 400.
    "lucidya-transcribe-v1 (aiva-stt)": ModelSetup(
        "lucidya-transcribe-v1", None, lambda model: qwen3_asr_own_template(model, "http://127.0.0.1:8008/v1")
    ),
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
RUN_MODELS = ["Cohere Transcribe Arabic (aiva-stt)", "lucidya-transcribe-v1 (aiva-stt)"]


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


if __name__ == "__main__":
    # Unless serve_args is None, serve first with: MODEL=<setup.model> bash vllm.sh <setup.serve_args>
    # NB: all datasets together are ~29 GB, more than the aiva GPU node's disks can spare (its watchdog
    # evicts every pod below 5% free), so there run from a tmpfs scratch dir with HF_HOME inside it;
    # data/ lands there too and drop_hub_downloads frees each wave's raw downloads:
    #   cd /tmp/<scratch> && HF_HOME=$PWD/hf HF_TOKEN=... uv run --project <repo> python <repo>/main.py
    results = Path(__file__).parent / "results"
    results.mkdir(exist_ok=True)
    cache_paths = {name: results / f"{slug(name)}.json" for name in RUN_MODELS}

    def load_caches() -> dict[str, dict[str, list[float]]]:
        return {name: json.loads(path.read_text()) if path.exists() else {} for name, path in cache_paths.items()}

    # Only datasets some model has not scored yet are downloaded, so a finished run just prints its tables.
    unscored = {
        f"{benchmark}/{name}" for benchmark, datasets in BENCHMARKS.items() for name in datasets
        if any(f"{benchmark}/{name}" not in cache for cache in load_caches().values())
    }
    built = load_datasets(after_wave=drop_hub_downloads, keys=unscored)
    for name in RUN_MODELS:
        setup = MODELS[name]
        print(f"\n## {name}\n", flush=True)
        report = main(setup.make_transcriber(setup.model), cache_path=str(cache_paths[name]), built_datasets=built)
        (results / f"{slug(name)}.md").write_text(f"## {name}\n\n`{setup.model}`\n\n{report}\n")
    comparison = format_comparison(load_caches())
    (results / "comparison.md").write_text(comparison + "\n")
    print(f"\n{comparison}")
