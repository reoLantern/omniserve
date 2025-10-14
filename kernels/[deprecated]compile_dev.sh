#!/usr/bin/env bash

# note: run conda activate first

export USE_NVCC_CCACHE=1
export CCACHE_CPP2=1
export CCACHE_BASEDIR="$PWD"
export CUDAHOSTCXX="$(command -v g++)"


set -euo pipefail

# usage: CLEAN=1 bash compile_dev.sh
if [[ "${CLEAN:-0}" == "1" ]]; then
  pip uninstall -y omniserve_backend
  python setup.py clean --all || true
  rm -rf build/ omniserve_backend/ .cuda_ccache *.egg-info
fi

# ------- enable ccache -------
if command -v ccache >/dev/null 2>&1; then
  for d in /usr/lib/ccache /usr/local/opt/ccache/libexec /opt/homebrew/opt/ccache/libexec; do
    [[ -d "$d" ]] && export PATH="$d:$PATH"
  done
  # 放大缓存，提升命中率
  ccache -M 20G || true
  ccache -s || true
fi

# export MAX_JOBS=${MAX_JOBS:-$(nproc)}
export MAX_JOBS=32
# export TORCH_CUDA_ARCH_LIST="8.0"   # A100

mkdir -p omniserve_backend

if ! pip show omniserve_backend >/dev/null 2>&1; then
  pip install -v -e . --no-build-isolation --config-settings editable_mode=compat
fi

# 只编译已改动的文件；可用 BUILD_EXT 过滤目标
# 例如：BUILD_EXT=fused_attention_per_tensor_dense
python setup.py build_ext --inplace

# 如确实需要安装到site-packages（非必须）：
# pip install -v --no-build-isolation .
