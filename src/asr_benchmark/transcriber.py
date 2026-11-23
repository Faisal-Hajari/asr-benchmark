from asr_benchmark.audio import Audio
from asr_benchmark.transcription_client import HumainFastTranscriptionClient, VLLMChatCompletionsClient, VLLMTranscriptionsClient
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Transcribtion:
    audio: Audio
    text: str
    inference_time_s: float


class Transcriber:
    def transcribe(self, audio: list[Audio]) -> list[Transcribtion]:
        raise NotImplementedError


class LiteLLMTranscriber(Transcriber):
    def __init__(
        self,
        client: VLLMTranscriptionsClient | VLLMChatCompletionsClient | HumainFastTranscriptionClient,
        language: str | None = None,
        transform: Callable[[Audio], Audio] | None = None,
        text_transform: Callable[[str], str] | None = None,
    ):
        self.client = client
        self.language = language
        self.transform = transform
        self.text_transform = text_transform

    def transcribe(self, audio: list[Audio]) -> list[Transcribtion]:
        # Transcriptions keep the original clips; only what is sent to the client is transformed.
        sent = [self.transform(clip) for clip in audio] if self.transform else audio
        results = self.client.transcribe(sent, [self.language] * len(sent))
        if self.text_transform:
            results = [(self.text_transform(text), seconds) for text, seconds in results]
        return [Transcribtion(clip, text, seconds) for clip, (text, seconds) in zip(audio, results)]
