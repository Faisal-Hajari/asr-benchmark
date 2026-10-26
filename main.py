from asr_benchmark.datasets import DACS, ESCWA, MASC, MGB2, MGB3, SADA, Casablanca, CommonVoice
from asr_benchmark.evaluation import evaluate
from asr_benchmark.transcriber import Transcriber, LiteLLMTranscriber
from asr_benchmark.transcription_client import QWEN3_ASR_CHAT_TEMPLATE, VLLMChatCompletionsClient, VLLMTranscriptionsClient
from asr_benchmark.transforms import qwen_asr_postprocessing, qwen_asr_preprocessing

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
    },
    "Code Switch AR-EN": {
        "ESCWA": ESCWA,
    },
}
# Only the leaderboard's six test sets are averaged, matching how the leaderboard reports its average.
AVERAGED = {"Leaderboard"}
# (row label, scale, format) for each value evaluate returns, in order.
METRICS = [("WER %", 100, ".2f"), ("CER %", 100, ".2f"), ("RTF", 1, ".4f"), ("Latency s", 1, ".3f"), ("Audio min/s", 1, ".2f")]
NUM_WORKERS = 16
BATCH_SIZE = 16
LANG= "ar"
MAX_NUM_TOKEN=256

def main(
    transcriber: Transcriber
) -> None:
    results = {
        benchmark: {name: evaluate(transcriber, make_dataset(), batch_size=BATCH_SIZE) for name, make_dataset in datasets.items()}
        for benchmark, datasets in BENCHMARKS.items()
    }
    for benchmark, scores in results.items():
        if benchmark in AVERAGED:
            scores = {"Avg": tuple(sum(column) / len(column) for column in zip(*scores.values())), **scores}
        print_table(benchmark, scores)


def print_table(title: str, scores: dict[str, tuple[float, float, float, float, float]]) -> None:
    """Prints a markdown table with one column per dataset and one row per metric."""
    names = ["Metric", *scores]
    widths = [max(len(name), 11) for name in names]
    rows = [names, ["-" * width for width in widths]]
    for index, (label, scale, fmt) in enumerate(METRICS):
        rows.append([label, *(format(values[index] * scale, fmt) for values in scores.values())])
    print(f"\n{title}")
    for row in rows:
        print("| " + " | ".join(cell.rjust(width) for cell, width in zip(row, widths)) + " |")


if __name__ == "__main__":
    #Cohere: 
    # model = "CohereLabs/cohere-transcribe-arabic-07-2026",
    # client = VLLMTranscriptionsClient(model, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN)
    # transcriber = LiteLLMTranscriber(client, language=LANG)
    # main(transcriber)
    
    # # Audar, reproducing its qwen-asr reference setup. Serve with:
    # model = "audarai/Audar-ASR-V1-Turbo"
    # client = VLLMChatCompletionsClient(
    #     model,
    #     max_workers=NUM_WORKERS,
    #     max_num_tokens=MAX_NUM_TOKEN,
    #     chat_template=QWEN3_ASR_CHAT_TEMPLATE,
    #     language_names={"ar": "Arabic"},
    #     wav_subtype="FLOAT",
    # )
    # transcriber = LiteLLMTranscriber(
    #     client, language=LANG, transform=qwen_asr_preprocessing(), text_transform=qwen_asr_postprocessing()
    # )
    # main(transcriber)

    # Voxtral Mini, with the model card's recommended vLLM setup (transcription endpoint, temperature 0,
    # which is the endpoint's default). Serve with:
    #   MODEL=mistralai/Voxtral-Mini-3B-2507 GPU_MEMORY_UTILIZATION=0.89 bash vllm.sh --tokenizer_mode mistral \
    #     --config_format mistral --load_format mistral --max-model-len 2048 --max-num-seqs 16 --enforce-eager
    model = "mistralai/Voxtral-Mini-3B-2507"
    client = VLLMTranscriptionsClient(model, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN)
    transcriber = LiteLLMTranscriber(client, language=LANG)
    main(transcriber)
