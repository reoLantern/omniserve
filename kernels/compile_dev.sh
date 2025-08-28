#!/usr/bin/env bash

# note: run conda activate first


set -euo pipefail

# ------- 可选：只在需要彻底重建时才清理 -------
# usage: CLEAN=1 bash compile_dev.sh
if [[ "${CLEAN:-0}" == "1" ]]; then
  python setup.py clean --all || true
  rm -rf build/ *.egg-info
fi

# ------- 开启 ccache（正确方式） -------
if command -v ccache >/dev/null 2>&1; then
  for d in /usr/lib/ccache /usr/local/opt/ccache/libexec /opt/homebrew/opt/ccache/libexec; do
    [[ -d "$d" ]] && export PATH="$d:$PATH"
  done
  # 放大缓存，提升命中率
  ccache -M 20G || true
  ccache -s || true
fi

export MAX_JOBS=${MAX_JOBS:-$(nproc)}
# 开发期强烈建议固定架构，避免 fatbin 过多导致慢编译
# export TORCH_CUDA_ARCH_LIST="8.0"   # A100

mkdir -p omniserve_backend

# ------- 开发循环（增量构建） -------
# 第一次或环境变了：做一次可编辑安装，之后无需重复
if ! pip show omniserve_backend >/dev/null 2>&1; then
  pip install -e .
fi

# 只编译已改动的文件；可用 BUILD_EXT 过滤目标
# 例如：BUILD_EXT=fused_attention_per_tensor_dense
python setup.py build_ext --inplace -j "$MAX_JOBS"

# 如确实需要安装到site-packages（非必须）：
# pip install -v --no-build-isolation .
