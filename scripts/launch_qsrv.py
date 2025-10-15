#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, runpy, argparse, subprocess
from datetime import datetime
from pathlib import Path

def expand_path(p: str) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(p)))

# ==== CLI ====
parser = argparse.ArgumentParser(description="Launch qserve_benchmark with CLI-only config.")
parser.add_argument("--prompt-len",        type=int, default=512)
parser.add_argument("--decode-len",        type=int, default=512)
parser.add_argument("--global-batch-size", type=int, default=32)
parser.add_argument("--model-path",        type=str, default=None)
parser.add_argument("--json-prefix",       type=str, default="kernel_calls")
parser.add_argument("--precision",         type=str, default="w4a8kv4")
parser.add_argument("--group-size",        type=str, default="128")
parser.add_argument("--dump-dir",          type=str, default=None)
parser.add_argument("--enable-call-logger",type=int, choices=[0,1], default=1)
parser.add_argument("--trace-pybind",      type=int, choices=[0,1], default=0)
args, passthrough = parser.parse_known_args()

# ==== workspace & defaults ====
scripts_dir = Path(__file__).resolve().parent
ws = scripts_dir.parent          # 仓库根目录（omniserve）
os.chdir(ws)

default_model_dir = ws / "QServe-benchmarks" / "Llama-3-8B"
model_path = expand_path(args.model_path) if args.model_path else default_model_dir

dump_dir = expand_path(args.dump_dir) if args.dump_dir else (ws / "dump")
dump_dir.mkdir(parents=True, exist_ok=True)

# 如默认模型目录不存在，尝试克隆
if model_path == default_model_dir and not default_model_dir.exists():
    repo_root = ws / "QServe-benchmarks"
    if not repo_root.exists():
        try:
            subprocess.run(
                ["git", "clone", "https://www.modelscope.cn/datasets/mit-han-lab/QServe-benchmarks"],
                cwd=str(ws), check=False
            )
        except Exception:
            pass

# ==== NUM_GPU_PAGE_BLOCKS = batch_size * ( ceil((P+D)/64) + 1 ) ====
tokens_total = args.prompt_len + args.decode_len
num_gpu_page_blocks = args.global_batch_size * ((tokens_total + 63) // 64 + 1)

# ==== dump file name ====
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
dump_file = dump_dir / f"{args.json_prefix}_{args.prompt_len}p_{args.decode_len}d_bs{args.global_batch_size}_{model_path.name}_{ts}.jsonl"

# ==== 导出供底层使用的环境变量 ====
os.environ["QSRV_ENABLE_CALL_LOGGER"] = str(args.enable_call_logger)
os.environ["QSRV_TRACE_PYBIND"] = str(args.trace_pybind)
os.environ["GLOBAL_BATCH_SIZE"] = str(args.global_batch_size)
os.environ["NUM_GPU_PAGE_BLOCKS"] = str(num_gpu_page_blocks)
os.environ["QSRV_DUMP_KERNEL_FILE"] = str(dump_file)
os.environ["MODEL_PATH"] = str(model_path)

# ==== 组装并运行 ====
argv = [
    "qserve_benchmark.py",
    "--model", str(model_path),
    "--benchmarking",
    "--precision", args.precision,
    "--group-size", args.group_size,
    "--prompt-len", str(args.prompt_len),
    "--generation-len", str(args.decode_len),
    *passthrough,   # 允许在 .sh 或 launch.json 里追加透传参数
]

print("[qsrv-launch] cwd:", ws)
print("[qsrv-launch] MODEL_PATH:", model_path)
print("[qsrv-launch] P/D/GBS:", args.prompt_len, args.decode_len, args.global_batch_size)
print("[qsrv-launch] NUM_GPU_PAGE_BLOCKS:", num_gpu_page_blocks)
print("[qsrv-launch] DUMP_FILE:", dump_file)
print("[qsrv-launch] argv:", " ".join(argv))

sys.argv = argv
runpy.run_path("qserve_benchmark.py", run_name="__main__")
