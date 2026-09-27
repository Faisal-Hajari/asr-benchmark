import base64
import time
from concurrent.futures import ThreadPoolExecutor

import litellm

from asr_benchmark.audio import Audio, to_wav_bytes


class VLLMTranscriptionsClient:
    """Transcribes through a vLLM server's /v1/audio/transcriptions endpoint.

    Args:
        model: The model name the server was started with.
        api_base: OpenAI-style base URL including /v1; LiteLLM appends /audio/transcriptions.
        max_workers: Concurrent requests per transcribe call.
        max_num_tokens: Cap on generated tokens per clip, sent as vLLM's max_completion_tokens.
    """

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        max_workers: int = 8,
        max_num_tokens: int | None = None,
    ):
        self.model = model
        self.api_base = api_base
        self.max_workers = max_workers
        self.max_num_tokens = max_num_tokens

    def _transcribe_one(self, audio: Audio, language: str | None = None) -> tuple[str, float]:
        start = time.perf_counter()
        response = litellm.transcription(
            model=f"hosted_vllm/{self.model}",
            file=("audio.wav", to_wav_bytes(audio)),
            api_base=self.api_base,
            language=language,
            max_completion_tokens=self.max_num_tokens,
        )
        return response.text, time.perf_counter() - start

    def transcribe(self, audio: list[Audio], languages: list[str | None]) -> list[tuple[str, float]]:
        """Sends one request per clip and returns (text, seconds) per clip, timed per request.
        languages[i] is the ISO 639-1 code for audio[i]. 
        """
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._transcribe_one, audio, languages))


# Qwen3-ASR's prompt as qwen-asr builds it: an always-present (possibly empty) system turn, one audio,
# and, when a language is given, the assistant turn prefilled with "language {name}<asr_text>" so the
# model outputs text only. vLLM's transcription endpoint cannot produce the empty system turn.
QWEN3_ASR_CHAT_TEMPLATE = (
    "<|im_start|>system\n"
    "{% for message in messages if message.role == 'system' %}{{ message.content }}{% endfor %}"
    "<|im_end|>\n"
    "<|im_start|>user\n<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
    "<|im_start|>assistant\n"
    "{% if language %}language {{ language }}<asr_text>{% endif %}"
)


class VLLMChatCompletionsClient:
    """Transcribes through a vLLM server's /v1/chat/completions endpoint, for audio LLMs.

    The clip goes in as base64 input_audio. The system message and/or a custom chat template set
    the task and language, as in the Audar-ASR-V1 model card and the qwen-asr package.

    Args:
        model: The model name the server was started with.
        api_base: OpenAI-style base URL including /v1.
        max_workers: Concurrent requests per transcribe call.
        max_num_tokens: Cap on generated tokens per clip, sent as max_tokens.
        system_prompts: System message per ISO 639-1 language; a language missing here gets no system message.
        temperature: Sampling temperature; 0 decodes greedily.
        chat_template: Jinja template sent with every request, e.g. QWEN3_ASR_CHAT_TEMPLATE. The server
            must run with --trust-request-chat-template. None uses the model's own template.
        language_names: ISO 639-1 code to the name passed to chat_template as `language`, e.g.
            {"ar": "Arabic"}. A code missing here passes no language.
        wav_subtype: soundfile subtype the clip is encoded with; "FLOAT" avoids 16-bit quantization.
    """

    def __init__(
        self,
        model: str,
        api_base: str = "http://localhost:8000/v1",
        max_workers: int = 8,
        max_num_tokens: int | None = None,
        system_prompts: dict[str, str] | None = None,
        temperature: float = 0.0,
        chat_template: str | None = None,
        language_names: dict[str, str] | None = None,
        wav_subtype: str = "PCM_16",
    ):
        self.model = model
        self.api_base = api_base
        self.max_workers = max_workers
        self.max_num_tokens = max_num_tokens
        self.system_prompts = system_prompts or {}
        self.temperature = temperature
        self.chat_template = chat_template
        self.language_names = language_names or {}
        self.wav_subtype = wav_subtype

    def _transcribe_one(self, audio: Audio, language: str | None = None) -> tuple[str, float]:
        start = time.perf_counter()
        wav = base64.b64encode(to_wav_bytes(audio, subtype=self.wav_subtype)).decode()
        system_prompt = self.system_prompts.get(language)
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append(
            {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": wav, "format": "wav"}}]}
        )
        extra_body = {}
        if self.chat_template is not None:
            extra_body = {
                "chat_template": self.chat_template,
                "chat_template_kwargs": {"language": self.language_names.get(language)},
            }
        response = litellm.completion(
            model=f"hosted_vllm/{self.model}",
            messages=messages,
            api_base=self.api_base,
            max_tokens=self.max_num_tokens,
            temperature=self.temperature,
            extra_body=extra_body,
        )
        return response.choices[0].message.content, time.perf_counter() - start

    def transcribe(self, audio: list[Audio], languages: list[str | None]) -> list[tuple[str, float]]:
        """Sends one request per clip and returns (text, seconds) per clip, timed per request.

        languages[i] is the ISO 639-1 code for audio[i].
        """
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._transcribe_one, audio, languages))

