#!/bin/bash
set -euo pipefail

HF_MODEL_ID="${HF_MODEL_ID:-unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit}"
GGUF_PATH="${GGUF_PATH:-/models/qwen2.5-3b-instruct.gguf}"
HF_DIR="${HF_DIR:-/models/hf_source}"
if [[ -z "${CONVERT_SCRIPT:-}" ]]; then
  CONVERT_SCRIPT="$(find /app -name 'convert_hf_to_gguf.py' 2>/dev/null | head -1)"
fi
CONVERT_SCRIPT="${CONVERT_SCRIPT:-/app/convert_hf_to_gguf.py}"

mkdir -p /models "$(dirname "$GGUF_PATH")"

if [[ -f "$GGUF_PATH" ]]; then
  echo "GGUF already present: $GGUF_PATH"
  exit 0
fi

# Optional: download a ready-made GGUF instead of converting safetensors
if [[ -n "${HF_GGUF_REPO:-}" && -n "${HF_GGUF_FILE:-}" ]]; then
  echo "Downloading GGUF ${HF_GGUF_REPO}/${HF_GGUF_FILE} ..."
  pip install -q --no-cache-dir "huggingface_hub[cli]"
  huggingface-cli download "$HF_GGUF_REPO" "$HF_GGUF_FILE" \
    --local-dir /models/gguf_download \
    ${HF_TOKEN:+--token "$HF_TOKEN"}
  cp "/models/gguf_download/${HF_GGUF_FILE}" "$GGUF_PATH"
  echo "GGUF ready at $GGUF_PATH"
  exit 0
fi

echo "Downloading ${HF_MODEL_ID} (this may take a while) ..."
pip install -q --no-cache-dir "huggingface_hub[cli]"
huggingface-cli download "$HF_MODEL_ID" \
  --local-dir "$HF_DIR" \
  ${HF_TOKEN:+--token "$HF_TOKEN"}

if [[ ! -f "$CONVERT_SCRIPT" ]]; then
  echo "ERROR: convert_hf_to_gguf.py not found at $CONVERT_SCRIPT"
  echo "Set HF_GGUF_REPO and HF_GGUF_FILE in .env to use a pre-built GGUF."
  exit 1
fi

echo "Converting to GGUF: $GGUF_PATH"
python "$CONVERT_SCRIPT" "$HF_DIR" --outfile "$GGUF_PATH" --outtype f16

if [[ ! -f "$GGUF_PATH" ]]; then
  echo "ERROR: conversion did not produce $GGUF_PATH"
  exit 1
fi

echo "Model ready: $GGUF_PATH"
