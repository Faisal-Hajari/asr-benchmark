import dataclasses
import math
import re
from collections.abc import Callable

import numpy as np
import soxr

from asr_benchmark.audio import Audio
from asr_benchmark.voxtral_normalization import normalize_arabic_asr


class AudioTransform:
    def __call__(self, audio: Audio) -> Audio:
        raise NotImplementedError


class TextTransform:
    def __call__(self, text: str) -> str:
        raise NotImplementedError


class Compose[T]:
    """Chains transforms of one kind: AudioTransforms on Audio or TextTransforms on text."""

    def __init__(self, transforms: list[Callable[[T], T]]):
        self.transforms = transforms

    def __call__(self, value: T) -> T:
        for transform in self.transforms:
            value = transform(value)
        return value


class ToMono(AudioTransform):
    def __call__(self, audio: Audio) -> Audio:
        return dataclasses.replace(audio, samples=audio.samples.mean(axis=1, keepdims=True, dtype=np.float32))


class Resample(AudioTransform):
    """Resamples with soxr; quality="HQ" is what librosa.resample uses by default."""

    def __init__(self, sample_rate: int = 16000, quality: str = "HQ"):
        self.sample_rate = sample_rate
        self.quality = quality

    def __call__(self, audio: Audio) -> Audio:
        if audio.sample_rate == self.sample_rate:
            return audio
        samples = soxr.resample(audio.samples, audio.sample_rate, self.sample_rate, quality=self.quality)
        # NB: like librosa.resample, pad or trim to ceil(n * ratio); soxr alone can return one sample less.
        length = math.ceil(audio.samples.shape[0] * (self.sample_rate / audio.sample_rate))
        samples = np.pad(samples[:length], ((0, max(0, length - samples.shape[0])), (0, 0)))
        return dataclasses.replace(audio, samples=samples.astype(np.float32), sample_rate=self.sample_rate)


class PeakNormalize(AudioTransform):
    """Scales audio down to a peak of 1.0 when it exceeds it; audio already in range is untouched."""

    def __call__(self, audio: Audio) -> Audio:
        peak = float(np.abs(audio.samples).max(initial=0.0))
        if peak <= 1.0:
            return audio
        return dataclasses.replace(audio, samples=audio.samples / np.float32(peak))


class PadToMinDuration(AudioTransform):
    """Zero-pads the end of clips shorter than min_duration_s."""

    def __init__(self, min_duration_s: float = 0.5):
        self.min_duration_s = min_duration_s

    def __call__(self, audio: Audio) -> Audio:
        missing = int(self.min_duration_s * audio.sample_rate) - audio.samples.shape[0]
        if missing <= 0:
            return audio
        return dataclasses.replace(audio, samples=np.pad(audio.samples, ((0, missing), (0, 0))))


class Strip(TextTransform):
    def __call__(self, text: str) -> str:
        return text.strip()


class FixRepetitions(TextTransform):
    """Collapses runaway repetitions in decoded text.

    Port of detect_and_fix_repetitions from qwen-asr 0.0.6 (Apache-2.0): a character repeated more
    than `threshold` times is kept once, and the first pattern of up to `max_pattern_len` characters
    repeated `threshold` or more times is kept once.
    """

    def __init__(self, threshold: int = 20, max_pattern_len: int = 20):
        self.threshold = threshold
        self.max_pattern_len = max_pattern_len

    def __call__(self, text: str) -> str:
        return self._collapse_patterns(self._collapse_chars(text))

    def _collapse_chars(self, text: str) -> str:
        pieces = []
        i = 0
        while i < len(text):
            count = 1
            while i + count < len(text) and text[i + count] == text[i]:
                count += 1
            pieces.append(text[i] if count > self.threshold else text[i : i + count])
            i += count
        return "".join(pieces)

    def _collapse_patterns(self, text: str) -> str:
        # NB: mirrors the reference exactly, including that it recurses only into the text after the
        # first collapsed run and that a string shorter than 2 * threshold is returned unchanged.
        n = len(text)
        if n < self.threshold * 2:
            return text
        pieces = []
        i = 0
        while i <= n - self.threshold * 2:
            for k in range(1, self.max_pattern_len + 1):
                if i + k * self.threshold > n:
                    break
                pattern = text[i : i + k]
                if all(text[i + rep * k : i + rep * k + k] == pattern for rep in range(1, self.threshold)):
                    end = i + self.threshold * k
                    while end + k <= n and text[end : end + k] == pattern:
                        end += k
                    pieces.append(pattern)
                    pieces.append(self._collapse_patterns(text[end:]))
                    return "".join(pieces)
            pieces.append(text[i])
            i += 1
        pieces.append(text[i:])
        return "".join(pieces)


class SplitArabicLatin(TextTransform):
    """Puts a space where Arabic and Latin letters touch, e.g. الpodcast -> ال podcast."""

    BOUNDARY = re.compile(r"(?<=[\u0600-\u06FF])(?=[A-Za-z])|(?<=[A-Za-z])(?=[\u0600-\u06FF])")

    def __call__(self, text: str) -> str:
        return self.BOUNDARY.sub(" ", text)


class Lowercase(TextTransform):
    """Lowercases Latin text; Arabic has no case and is left as it is."""

    def __call__(self, text: str) -> str:
        return text.lower()


class TaMarbutaToHa(TextTransform):
    """Writes ة as ه, so a final ta marbuta and ha, which sound the same there, count as one spelling."""

    def __call__(self, text: str) -> str:
        return text.replace("ة", "ه")


def spelling_postprocessing() -> Compose[str]:
    """Prediction post-processing for the spelling differences plain WER counts as errors:
    a prefix glued to an English word, English case, and ة versus ه."""
    return Compose([SplitArabicLatin(), Lowercase(), TaMarbutaToHa()])


class VoxtralArabicNormalize(TextTransform):
    """Mistral's Arabic ASR normalization from the Voxtral-Mini-4B-Realtime-Arabic card.

    It lowercases, removes diacritics and punctuation, writes ذ as د and ث as ت, splits clitics,
    maps dialect and loanword spellings and turns spoken numbers into digits. Here it runs on model
    output only (voxtral_postprocessing); references keep the benchmark's own normalization.
    """

    def __call__(self, text: str) -> str:
        return normalize_arabic_asr(text)


def voxtral_postprocessing() -> Compose[str]:
    """Prediction post-processing for Voxtral: strip the leading space the transcription endpoint
    adds, then Mistral's Arabic normalization."""
    return Compose([Strip(), VoxtralArabicNormalize()])


def qwen_asr_preprocessing() -> Compose[Audio]:
    """The audio preprocessing qwen-asr 0.0.6 applies before inference (normalize_audio_input and
    the MIN_ASR_INPUT_SECONDS padding in split_audio_into_chunks)."""
    return Compose([ToMono(), Resample(16000), PeakNormalize(), PadToMinDuration(0.5)])


def qwen_asr_postprocessing() -> Compose[str]:
    """The text post-processing qwen-asr 0.0.6 applies to a forced-language output (parse_asr_output)."""
    return Compose([Strip(), FixRepetitions()])
