from concurrent.futures import ThreadPoolExecutor

import litellm

from asr_benchmark.audio import Audio, to_wav_bytes


class VLLMTranscriptionsClient:
    """Transcribes through a vLLM server's /v1/audio/transcriptions endpoint.

    Args:
        model: The model name the server was started with.
        api_base: OpenAI-style base URL including /v1; LiteLLM appends /audio/transcriptions.
        max_workers: Concurrent requests per transcribe call.
    """

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        max_workers: int = 8,
    ):
        self.model = model
        self.api_base = api_base
        self.max_workers = max_workers

    def _transcribe_one(self, audio: Audio, language: str | None = None) -> str:
        response = litellm.transcription(
            model=f"hosted_vllm/{self.model}",
            file=("audio.wav", to_wav_bytes(audio)),
            api_base=self.api_base,
            language=language,
        )
        return response.text

    def transcribe(self, audio: list[Audio], languages: list[str | None]) -> list[str]:
        """Sends one request per clip; languages[i] is the ISO 639-1 code for audio[i], None lets the model decide."""
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._transcribe_one, audio, languages))
