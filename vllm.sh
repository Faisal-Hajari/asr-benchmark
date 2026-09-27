#!/usr/bin/env bash
# Serve Cohere Transcribe Arabic (Arabic, Arabic dialects, English, code-switched) with vLLM.
# Settings from https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026

MODEL="${MODEL:-CohereLabs/cohere-transcribe-arabic-07-2026}"
VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

# NB: the image ships vllm without the audio extra, so it is installed on every start. The extra is
# pinned to the image's own vllm version; any other pin reinstalls vllm and downgrades torch/transformers.
exec docker run --rm --gpus all --ipc=host -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash "vllm/vllm-openai:v$VLLM_VERSION" \
  -c 'uv pip install --system "vllm[audio]==$0" librosa && exec vllm serve "$1" --trust-remote-code --gpu-memory-utilization "$2" "${@:3}"' \
  "$VLLM_VERSION" "$MODEL" "$GPU_MEMORY_UTILIZATION" "$@"
