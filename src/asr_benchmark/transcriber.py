from asr_benchmark.audio import Audio
from asr_benchmark.transcription_client import VLLMTranscriptionsClient

class Transcriber:
    def transcribe(self, audio: list[Audio]) -> list[str]:
        raise NotImplementedError


class LiteLLMTranscriber(Transcriber):
    def __init__(self, client:VLLMTranscriptionsClient, language:str | None= None):
        self.client = client
        self.language = language

    def transcribe(self, audio: list[Audio]) -> list[str]:
        return self.client.transcribe(audio, [self.language] * len(audio))
