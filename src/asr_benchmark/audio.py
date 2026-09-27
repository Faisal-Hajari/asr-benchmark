from dataclasses import dataclass

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32 in [-1, 1], shape (num_samples, channels)
    sample_rate: int


def load(path: str, dtype:str="float32", always_2d:bool=True) -> Audio:
    samples, sample_rate = sf.read(path, dtype=dtype, always_2d=always_2d)
    return Audio(samples, sample_rate)
