import dataclasses
import os
import time
from collections.abc import Callable

# NB: must be set before torch initializes CUDA; a growing encoder cache fragments memory otherwise.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402
from transformers import AutoProcessor, VoxtralRealtimeForConditionalGeneration  # noqa: E402

from asr_benchmark.audio import Audio  # noqa: E402
from asr_benchmark.transcriber import Transcriber, Transcribtion  # noqa: E402
from asr_benchmark.transforms import Compose, Resample, ToMono  # noqa: E402


class VoxtralRealtimeTranscriber(Transcriber):
    """Runs Voxtral Realtime in this process with transformers, many clips per generate call.

    vLLM 0.30 mixes text from one clip into another when it serves this model concurrent requests;
    transformers' batched generate returns the same text as one clip at a time.

    Args:
        model: Hugging Face model id.
        transform: Applied to each clip before inference; None converts to the 16 kHz mono the
            model expects.
        text_transform: Applied to each decoded text.
        max_batch_audio_s: Audio per generate call, counted as clips times the longest clip since
            a batch is padded to its longest clip. The encoder cache grows with it, ~28 MB per second
            in BF16, so 60 s is what a 12 GB card holds next to the 8.3 GB of weights.
        max_clip_s: Clips longer than this are split into consecutive pieces of at most this
            length, transcribed separately and joined.
        device: Torch device the model runs on.
    """

    def __init__(
        self,
        model: str,
        transform: Callable[[Audio], Audio] | None = None,
        text_transform: Callable[[str], str] | None = None,
        max_batch_audio_s: float = 60.0,
        max_clip_s: float = 60.0,
        device: str = "cuda",
    ):
        self.processor = AutoProcessor.from_pretrained(model)
        self.model = VoxtralRealtimeForConditionalGeneration.from_pretrained(model, dtype=torch.bfloat16).to(device).eval()
        self.transform = transform or Compose([ToMono(), Resample(16000)])
        self.text_transform = text_transform
        self.max_batch_audio_s = max_batch_audio_s
        self.max_clip_s = max_clip_s
        self.device = device

    def transcribe(self, audio: list[Audio]) -> list[Transcribtion]:
        # Each piece is (its clip's index, its position in the clip, audio); clips up to max_clip_s
        # are one piece.
        pieces = [
            (i, position, piece)
            for i, clip in enumerate(audio)
            for position, piece in enumerate(self._split(self.transform(clip)))
        ]
        texts: list[dict[int, str]] = [{} for _ in audio]
        seconds = [0.0] * len(audio)
        for batch in self._batches([piece for *_, piece in pieces]):
            start = time.perf_counter()
            inputs = self.processor([pieces[j][2].samples[:, 0] for j in batch], return_tensors="pt")
            inputs = inputs.to(self.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, do_sample=False)
            decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)
            # NB: every clip in a batch gets the batch's time, as concurrent vLLM requests each do.
            elapsed = time.perf_counter() - start
            for j, text in zip(batch, decoded):
                i, position, _ = pieces[j]
                texts[i][position] = text
                seconds[i] += elapsed
        results = []
        for clip, clip_texts, clip_seconds in zip(audio, texts, seconds):
            # NB: pieces are batched by length, which puts a clip's shorter last piece first.
            text = " ".join(clip_texts[position].strip() for position in sorted(clip_texts))
            results.append(Transcribtion(clip, self.text_transform(text) if self.text_transform else text, clip_seconds))
        return results

    def _split(self, clip: Audio) -> list[Audio]:
        # ponytail: fixed-length pieces can cut a word in two; split at silences if long clips matter.
        size = int(self.max_clip_s * clip.sample_rate)
        return [dataclasses.replace(clip, samples=clip.samples[i : i + size]) for i in range(0, clip.samples.shape[0], size)]

    def _batches(self, sent: list[Audio]) -> list[list[int]]:
        """Groups clip indices by length so padding stays within max_batch_audio_s."""
        batches = []
        for i in sorted(range(len(sent)), key=lambda i: sent[i].audio_length_s):
            # Sorted ascending, so the clip being added is the batch's longest.
            if batches and (len(batches[-1]) + 1) * sent[i].audio_length_s <= self.max_batch_audio_s:
                batches[-1].append(i)
            else:
                batches.append([i])
        return batches
