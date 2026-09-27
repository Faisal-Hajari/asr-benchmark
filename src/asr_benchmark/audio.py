import io
from dataclasses import dataclass
from typing import BinaryIO, Optional

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32 in [-1, 1], shape (num_samples, channels)
    sample_rate: int
    transcription: Optional[str] = None

    @property
    def audio_length_s(self) -> float:
        return self.samples.shape[0] / self.sample_rate


def load(path: str | BinaryIO, dtype:str="float32", always_2d:bool=True) -> Audio:
    samples, sample_rate = sf.read(path, dtype=dtype, always_2d=always_2d)
    return Audio(samples, sample_rate)


def to_wav_bytes(audio: Audio, subtype: str = "PCM_16", audio_format:str ="WAV") -> bytes:
    buffer = io.BytesIO()
    # NB: clip so out-of-range floats saturate instead of wrapping around in integer PCM.
    sf.write(buffer, np.clip(audio.samples, -1, 1), audio.sample_rate, format=audio_format, subtype=subtype)
    return buffer.getvalue()