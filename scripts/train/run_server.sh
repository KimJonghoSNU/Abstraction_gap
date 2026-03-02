#!/usr/bin/env bash
set -euo pipefail

# Intent: allow selecting a free GPU at runtime instead of hard-coding a potentially busy device.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
MODEL_PATH="${MODEL_PATH:-/data2/da02/models/Qwen3-4B-Instruct-2507}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9000}"
# Intent: avoid collision with TRL GRPO fixed communicator port(51216) by reserving a different base internal vLLM port range.
export VLLM_PORT="${VLLM_PORT:-27000}"

echo "[INFO] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[INFO] MODEL_PATH=${MODEL_PATH}"
echo "[INFO] PORT=${PORT}"
echo "[INFO] MAX_MODEL_LEN=${MAX_MODEL_LEN}"
echo "[INFO] VLLM_PORT=${VLLM_PORT}"
trl vllm-serve --model "${MODEL_PATH}" --max_model_len "${MAX_MODEL_LEN}" --port "${PORT}"
