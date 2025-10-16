#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

################################
PROMPT_LEN=1024
DECODE_LEN=512
GLOBAL_BATCH_SIZE=32
################################

ENABLE_CALL_LOGGER=1
MODEL_PATH="${ROOT_DIR}/QServe-benchmarks/Llama-3-8B"
# MODEL_PATH="${ROOT_DIR}/qserve_checkpoints/Llama-3-8B-Instruct-QServe-g128"

export CUDA_VISIBLE_DEVICES=1
export USE_RANDOM_TOKENS=1     # set to 1 to generate random tokens after layer to avoid bug

python "${SCRIPT_DIR}/launch_qsrv.py" \
  --prompt-len "${PROMPT_LEN}" \
  --decode-len "${DECODE_LEN}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --json-prefix "kernel_calls" \
  --precision "w4a8kv4" \
  --group-size "128" \
  --dump-dir "${ROOT_DIR}/dump" \
  --enable-call-logger "${ENABLE_CALL_LOGGER}" \
  --model-path "${MODEL_PATH}" \
  "$@"

# --precision: The precision for GEMM in QServe, please choose from the following values:
#              w4a8kv4, w4a8kv8, w4a8 (means w4a8kv8), w8a8kv4, w8a8kv8, w8a8 (means w8a8kv8). Default: w4a8kv4.
# --group-size: only -1 and 128 are supported. -1 means per-channel quantization, 128 means per-group quantization with group size 128
# --max-num-batched-tokens: Maximum number of batched tokens per iteration. Default: 262144
# --max-num-seqs: max batch size per iter. If GLOBAL_BATCH_SIZE ≤ max-num-seqs, then no affect. Default: 256