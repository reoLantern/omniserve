pip uninstall -y omniserve_backend
python setup.py clean --all || true
rm -rf build/ *.egg-info

export MAX_JOBS=$(nproc)           # ninja 的并行度
# export TORCH_CUDA_ARCH_LIST="8.0"  # 只编 A100

pip install -v --no-build-isolation .

# 只编译安装特定的扩展
# BUILD_EXT=replay_pd python setup.py install