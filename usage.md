创建环境：

```bash
cd omniserve
conda create -n QServe python=3.10 -y && conda activate QServe
pip install --upgrade pip
pip install -e .

conda config --env --set channel_priority strict
conda install -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
pip install flash-attn
# then test `import flash_attn` in python

# 如果 import flash_attn 失败，尝试下面的命令
pip uninstall -y flash-attn
export FLASH_ATTENTION_FORCE_BUILD=1
MAX_JOBS=32 pip install --no-build-isolation --no-cache-dir flash-attn

pip install ninja   # Install ninja if not already
cd kernels
# python setup.py install
bash compile.sh
```

下载模型（可选，benchmark 模式无需下载模型二进制文件）：

```bash
cd omniserve # 回到 omniserve 根目录
mkdir -p qserve_checkpoints && cd qserve_checkpoints
sudo apt-get install -y git-lfs
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-QServe
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mit-han-lab/Llama-3-8B-QServe
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/mit-han-lab/Llama-3-8B-Instruct-QServe-g128
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/mit-han-lab/Llama-3-8B-QServe
cd Llama-3-8B-Instruct-QServe
git lfs pull
```

运行 QServe：

```bash
cd omniserve  # 回到 omniserve 根目录
bash scripts/qserve_benchmark.sh
```

裁剪 dump 输出：

```bash
python3 ./dump/qserve_dump_filter.py ./dump/kernel_calls_1024p_16d_group128_bs4_Llama-3-8B_20251015_222924.jsonl --step 4
python3 ./dump/qserve_dump_filter.py ./dump/kernel_calls_1024p_512d_group-1_bs32_Llama-3-8B_20251015_225451.jsonl --step 50
```
