import base64
import contextlib
import hashlib
import json
import os
import queue
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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



class HumainFastTranscriptionClient:
    """Transcribes through HUMAIN Voice's fast transcription API (socket.io) with the humain-voice SDK.

    Each worker thread keeps one connection open for the whole call, so the time per clip covers
    upload and transcription but not connecting. A failed request is retried on a fresh connection
    with exponential backoff, e.g. after a rate limit.

    Every response is saved to cache_dir as <model>/<language>/<sha256 of the WAV sent>.json with its
    text, word timings and measured seconds, and a clip already there is not sent again: a rerun
    after a failure only pays for the clips it lacks, and keeps the original latency.

    Args:
        model: A humain_voice.stt.FastTranscriptionModel value, e.g. "bayan_cs_ar_en".
        api_url: Defaults to $HUMAIN_API_URL, else https://api.voice.humain.com.
        api_key: Defaults to $HUMAIN_API_KEY.
        max_workers: Concurrent connections per transcribe call.
        language: A humain_voice.stt.Language value ("ar", "en" or "codeswitch") sent for every clip,
            whatever language the benchmark passes: the model, not the clip, decides it.
        timeout_s: Per-clip timeout; Mixat's longest clip is 260 s.
        cache_dir: Where responses are saved; None disables the cache.
        max_attempts: Tries per clip before the error is raised.
    """

    def __init__(
        self,
        model: str,
        api_url: str | None = None,
        api_key: str | None = None,
        max_workers: int = 8,
        language: str = "codeswitch",
        timeout_s: float = 300.0,
        cache_dir: str | Path | None = "api_cache/humain",
        max_attempts: int = 10,
    ):
        self.model = model
        self.api_url = api_url or os.getenv("HUMAIN_API_URL", "https://api.voice.humain.com")
        self.api_key = api_key or os.environ["HUMAIN_API_KEY"]
        self.max_workers = max_workers
        self.language = language
        self.timeout_s = timeout_s
        self.cache_dir = Path(cache_dir) / model / language if cache_dir is not None else None
        self.max_attempts = max_attempts

    def _cache_path(self, wav: bytes) -> Path | None:
        return self.cache_dir / f"{hashlib.sha256(wav).hexdigest()}.json" if self.cache_dir else None

    def _save(self, path: Path, response, seconds: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": self.model,
            "language": self.language,
            "id": str(response.id),
            "transcription": response.transcription,
            "words": [{"word": w.word, "start_time": w.start_time, "end_time": w.end_time} for w in response.words or []],
            "seconds": seconds,
            "saved_at": time.time(),
        }
        # NB: written to a temp file and renamed so an interrupted run never leaves a half-written entry.
        temp = path.with_suffix(f".{threading.get_ident()}.tmp")
        temp.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
        temp.replace(path)

    def transcribe(self, audio: list[Audio], languages: list[str | None]) -> list[tuple[str, float]]:
        """Returns (text, seconds) per clip; languages is ignored, see the class docstring."""
        # NB: imported here so runs of other models do not need the SDK.
        from humain_voice import stt

        results: list[tuple[str, float] | None] = [None] * len(audio)
        pending = queue.SimpleQueue()
        for index, clip in enumerate(audio):
            wav = to_wav_bytes(clip)
            path = self._cache_path(wav)
            if path is not None and path.exists():
                record = json.loads(path.read_text(encoding="utf-8"))
                results[index] = (record["transcription"], record["seconds"])
            else:
                pending.put((index, wav, path))
        if pending.empty():
            return results

        def connect():
            client = stt.FastTranscriptionClient(api_url=self.api_url, api_key=self.api_key)
            client.__enter__()
            return client

        def work() -> None:
            client = None
            try:
                while True:
                    try:
                        index, wav, path = pending.get_nowait()
                    except queue.Empty:
                        return
                    for attempt in range(self.max_attempts):
                        try:
                            client = client or connect()
                            start = time.perf_counter()
                            response = client.transcribe_sync(
                                wav, self.language, self.model, timeout_seconds=self.timeout_s
                            )
                            seconds = time.perf_counter() - start
                            if response is None:
                                raise RuntimeError("no transcription returned")
                            break
                        except Exception as error:  # noqa: BLE001 - rate limits and drops surface as various types
                            # NB: retrying cannot fix an empty balance; stop now instead of after max_attempts.
                            if "insufficient credits" in str(error):
                                raise
                            if client is not None:
                                with contextlib.suppress(Exception):
                                    client.__exit__(None, None, None)
                                client = None
                            if attempt + 1 == self.max_attempts:
                                raise RuntimeError(f"HUMAIN failed clip {index} {self.max_attempts} times") from error
                            delay = min(60, 2 ** attempt) + random.random()
                            print(f"HUMAIN clip {index}: {error!r}; retrying in {delay:.0f} s", flush=True)
                            time.sleep(delay)
                    if path is not None:
                        self._save(path, response, seconds)
                    results[index] = (response.transcription, seconds)
            finally:
                if client is not None:
                    with contextlib.suppress(Exception):
                        client.__exit__(None, None, None)

        with ThreadPoolExecutor(self.max_workers) as pool:
            for future in [pool.submit(work) for _ in range(min(self.max_workers, pending.qsize()))]:
                future.result()
        return results
