#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"

PROMPT_LEN=2048
DECODE_LEN=512
GLOBAL_BATCH_SIZE=32

python "${SCRIPT_DIR}/launch_qsrv.py" \
  --prompt-len "${PROMPT_LEN}" \
  --decode-len "${DECODE_LEN}" \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --json-prefix "kernel_calls" \
  --precision "w4a8kv4" \
  --group-size "128" \
  --dump-dir "${ROOT_DIR}/dump" \
  --enable-call-logger 1 \
  "$@"
