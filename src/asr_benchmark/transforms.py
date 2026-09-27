from asr_benchmark.audio import Audio


class AudioTransform:
    def __call__(self, audio: Audio) -> Audio:
        raise NotImplementedError
