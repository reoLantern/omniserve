python qsr_replay.py \
  --jsonl ~/work/omniserve/dump/kernel_calls.jsonl \
  --module qgemm_w4a8_per_group \
  --name gemm_forward_cuda \
  --index 3 --show-launch
