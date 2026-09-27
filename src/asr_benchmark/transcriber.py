from asr_benchmark.audio import Audio


class Transcriber:
    def transcribe(self, audio: list[Audio]) -> list[str]:
        raise NotImplementedError
