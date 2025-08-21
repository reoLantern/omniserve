rm /home/mmy/work/omniserve/dump/kernel_calls.jsonl

MODEL_PATH=./qserve_checkpoints/Llama-3-8B-Instruct-QServe-g128

export QSRV_DUMP_KERNEL_CALLS=1
export QSRV_DUMP_KERNEL_FILE=/home/mmy/work/omniserve/dump/kernel_calls.jsonl
export QSRV_TRACE_PYBIND=1

# common_args="--max-num-batched-tokens 4195000 \
#              --chunk-prefill-size 1024000 \
#              --sparse-decode-mode 0"
common_args="--max-num-batched-tokens 419500 \
             --chunk-prefill-size 102400 \
             --sparse-decode-mode 0"

GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=$((25*GLOBAL_BATCH_SIZE)) \
NUM_RETRIEVAL_GPU_PAGE_BLOCKS=${NUM_GPU_PAGE_BLOCKS} \
NUM_STREAMING_GPU_PAGE_BLOCKS=0 \
CHUNK_PREFILL_SIZE=214700000 \
python qserve_benchmark.py \
  --model $MODEL_PATH \
  --benchmarking \
  --precision w4a8kv4 \
  --group-size 128 \
  --max-num-seqs 2 \
  --kv-quant-granularity fine_grained $common_args

# --max-num-batched-tokens: Maximum number of batched tokens per iteration. Default: 262144.
# --max-num-seqs 为 batch size