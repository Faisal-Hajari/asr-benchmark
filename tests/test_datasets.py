"""Every dataset must yield exactly the clips of the leaderboard manifest it replicates.

Manifests are copied from open_universal_arabic_asr_leaderboard/datasets at commit 8fce859.
"""

import json
from pathlib import Path

import pytest

from asr_benchmark.datasets import MASC, SADA, Casablanca

MANIFESTS = Path(__file__).parent / "manifests"

DATASETS = [
    (SADA, "sada_test.json"),
    (Casablanca, "casablanca_test.json"),
    (lambda: MASC("clean"), "masc_test_clean.json"),
    (lambda: MASC("noisy"), "masc_test_noisy.json"),
]

# NB: the SADA22 mirror cuts clip 6k_v2ms_SBA3_2426_0-seg_260_390-261_940 at 1.45 s; the manifest's
# segment timestamps say 1.55 s. The transcript is unique, so it is the right clip.
KNOWN_DURATION_MISMATCHES = {("sada_test.json", 1634)}


@pytest.mark.parametrize(("make_dataset", "manifest"), DATASETS, ids=[m for _, m in DATASETS])
def test_dataset_matches_leaderboard_manifest(make_dataset, manifest):
    items = [json.loads(line) for line in (MANIFESTS / manifest).read_text().splitlines()]
    dataset = make_dataset()

    assert len(dataset) == len(items)
    for index, item in enumerate(items):
        audio = dataset[index]
        assert audio.transcription == item["text"], index
        if (manifest, index) not in KNOWN_DURATION_MISMATCHES:
            assert audio.audio_length_s == pytest.approx(item["duration"], abs=0.01), index
