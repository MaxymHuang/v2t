#!/bin/bash
set -euo pipefail

GGUF_PATH="${GGUF_PATH:-/models/qwen2.5-3b-instruct.gguf}"
LLAMA_PORT="${LLAMA_PORT:-9990}"
LLAMA_CTX_SIZE="${LLAMA_CTX_SIZE:-8192}"
LLAMA_NGL="${LLAMA_NGL:-99}"
LLAMA_THREADS="${LLAMA_THREADS:-8}"

if [[ ! -f "$GGUF_PATH" ]]; then
  echo "ERROR: GGUF not found at $GGUF_PATH (run model-init first)"
  exit 1
fi

echo "Starting llama-server on port ${LLAMA_PORT} with ${GGUF_PATH}"
exec llama-server \
  -m "$GGUF_PATH" \
  --host 0.0.0.0 \
  --port "$LLAMA_PORT" \
  -c "$LLAMA_CTX_SIZE" \
  -ngl "$LLAMA_NGL" \
  --threads "$LLAMA_THREADS"
