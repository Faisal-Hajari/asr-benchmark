from asr_benchmark.datasets import MASC, SADA, Casablanca
from asr_benchmark.evaluation import evaluate
from asr_benchmark.transcriber import Transcriber, LiteLLMTranscriber
from asr_benchmark.transcription_client import QWEN3_ASR_CHAT_TEMPLATE, VLLMChatCompletionsClient, VLLMTranscriptionsClient
from asr_benchmark.transforms import qwen_asr_postprocessing, qwen_asr_preprocessing

# Built one at a time as the loop reaches them, so a failure in one does not cost loading the others.
DATASETS = {
    "SADA": SADA,
    "Casablanca": Casablanca,
    "MASC clean": lambda: MASC("clean"),
    "MASC noisy": lambda: MASC("noisy"),
}
NUM_WORKERS = 16
BATCH_SIZE = 16
LANG= "ar"
MAX_NUM_TOKEN=256

def main(
    transcriber: Transcriber
) -> None:
    results = {name: evaluate(transcriber, make_dataset(), batch_size=BATCH_SIZE) for name, make_dataset in DATASETS.items()}
    results["Average"] = tuple(sum(column) / len(column) for column in zip(*results.values()))
    print_table(results)


def print_table(results: dict[str, tuple[float, float, float, float, float]]) -> None:
    print(f"{'Dataset':<11} {'WER %':>7} {'CER %':>7} {'RTF':>7} {'Latency s':>10} {'Audio min/s':>12}")
    for name, (wer, cer, rtf, latency_s, throughput) in results.items():
        if name == "Average":
            print("-" * 58)
        print(f"{name:<11} {wer * 100:>7.2f} {cer * 100:>7.2f} {rtf:>7.4f} {latency_s:>10.3f} {throughput:>12.2f}")


if __name__ == "__main__":
    #Cohere: 
    # model = "CohereLabs/cohere-transcribe-arabic-07-2026",
    # client = VLLMTranscriptionsClient(model, max_workers=NUM_WORKERS, max_num_tokens=MAX_NUM_TOKEN)
    # transcriber = LiteLLMTranscriber(client, language=LANG)
    # main(transcriber)
    
    # Audar, reproducing its qwen-asr reference setup. Serve with:
    model = "audarai/Audar-ASR-V1-Turbo"
    client = VLLMChatCompletionsClient(
        model,
        max_workers=NUM_WORKERS,
        max_num_tokens=MAX_NUM_TOKEN,
        chat_template=QWEN3_ASR_CHAT_TEMPLATE,
        language_names={"ar": "Arabic"},
        wav_subtype="FLOAT",
    )
    transcriber = LiteLLMTranscriber(
        client, language=LANG, transform=qwen_asr_preprocessing(), text_transform=qwen_asr_postprocessing()
    )
    main(transcriber)
