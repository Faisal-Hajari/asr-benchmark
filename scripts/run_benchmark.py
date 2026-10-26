"""Benchmarks several models one after another, each on its own vLLM container.

For each model: stops any running vLLM container, serves the model with vllm.sh, smoke-tests two
clips of every dataset and runs main.main. Its section goes to results/<model>.md and each
dataset's metrics to results/<model>.json as they finish, so rerunning a model that failed resumes
it. results/benchmark.md is rebuilt from every model's section after each model.

Usage: uv run scripts/run_benchmark.py "Whisper large-v3" "Qwen3-ASR 0.6B" ...
"""

import datetime
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main  # noqa: E402

RESULTS = Path("results")
IMAGE = "vllm/vllm-openai:v0.30.0"


def stop_vllm() -> None:
    ids = subprocess.run(["docker", "ps", "-q", "--filter", f"ancestor={IMAGE}"], capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "stop", *ids], check=True, capture_output=True)


def wait_for(model: str, server: subprocess.Popen, timeout_s: float = 3600) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"vllm.sh exited with code {server.returncode}")
        try:
            served = [m["id"] for m in json.load(urllib.request.urlopen("http://localhost:8000/v1/models", timeout=5))["data"]]
            if model in served:
                return
        except OSError:
            pass
        time.sleep(10)
    raise TimeoutError(f"{model} was not served within {timeout_s:.0f} s")


def smoke_test(transcriber) -> None:
    for benchmark, datasets in main.BENCHMARKS.items():
        for name, make_dataset in datasets.items():
            dataset = make_dataset()
            outputs = transcriber.transcribe([dataset[0], dataset[1]])
            print(f"smoke {name}: {outputs[0].text[:60]!r}", flush=True)


def run(name: str) -> str:
    setup = main.MODELS[name]
    stop_vllm()
    if setup.serve_args is not None:
        with open(RESULTS / f"{slug(name)}.vllm.log", "w") as out:
            server = subprocess.Popen(["bash", "vllm.sh", *setup.serve_args], env={**os.environ, "MODEL": setup.model}, stdout=out, stderr=out)
    try:
        if setup.serve_args is not None:
            wait_for(setup.model, server)
        transcriber = setup.make_transcriber(setup.model)
        smoke_test(transcriber)
        return main.main(transcriber, cache_path=str(RESULTS / f"{slug(name)}.json"))
    finally:
        stop_vllm()


def slug(name: str) -> str:
    return name.lower().replace(" ", "-")


def write_report() -> None:
    header = (
        f"# ASR benchmark\n\nRun {datetime.date.today()} on vLLM 0.30.0 (`{IMAGE}`), "
        f"{main.NUM_WORKERS} concurrent requests in batches of {main.BATCH_SIZE}, language `{main.LANG}`, "
        f"at most {main.MAX_NUM_TOKEN} generated tokens per clip.\n\n"
        "The Leaderboard tables use the Open Universal Arabic ASR Leaderboard normalization, so they compare with "
        "published numbers (CommonVoice17 is the 17.0 test set; the leaderboard uses 18.0). The other tables "
        "add the spelling rules: a prefix glued to an English word is split off, English is lowercased and "
        "ة is written as ه.\n\n"
    )
    sections = [(RESULTS / f"{slug(name)}.md").read_text().strip() for name in main.MODELS if (RESULTS / f"{slug(name)}.md").exists()]
    (RESULTS / "benchmark.md").write_text(header + "\n\n".join(sections) + "\n")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        started = time.time()
        try:
            report = run(name)
        except Exception as error:  # noqa: BLE001 - one model failing must not cost the others' hours of runs
            report = f"**Failed:** `{type(error).__name__}: {error}`"
            print(f"{name} failed: {error!r}", flush=True)
        setup = main.MODELS[name]
        served = "run in this process with transformers" if setup.serve_args is None else f"served with `{' '.join(['vllm.sh', *setup.serve_args])}`"
        (RESULTS / f"{slug(name)}.md").write_text(
            f"## {name}\n\n`{setup.model}`, {served}. "
            f"Took {(time.time() - started) / 60:.0f} min.\n\n{report}\n"
        )
        write_report()
