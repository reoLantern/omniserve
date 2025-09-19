创建环境：

```bash
cd omniserve
conda create -n OmniServe python=3.10 -y && conda activate OmniServe
pip install --upgrade pip
pip install -e .

conda config --env --set channel_priority strict
conda install -c conda-forge "libstdcxx-ng>=13" "libgcc-ng>=13"
pip install flash-attn
# then test `import flash_attn` in python

wget https://github.com/mit-han-lab/Block-Sparse-Attention/releases/download/v0.0.1/block_sparse_attn-0.0.1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install block_sparse_attn-0.0.1+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# then test `import block_sparse_attn` in python

pip install ninja   # Install ninja if not already
cd kernels
python setup.py install     # 实验阶段，参考 kernels 文件夹的自定义 .sh 脚本
```

下载模型：

```bash
cd omniserve # 回到 omniserve 根目录
mkdir -p qserve_checkpoints && cd qserve_checkpoints
sudo apt-get install -y git-lfs
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mit-han-lab/Llama-3-8B-Instruct-QServe
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/mit-han-lab/Llama-3-8B-QServe
GIT_LFS_SKIP_SMUDGE=1 git clone https://www.modelscope.cn/mit-han-lab/Llama-3-8B-Instruct-QServe-g128
cd Llama-3-8B-Instruct-QServe
git lfs pull
```

使用 QServe 进行推理：

```bash
cd ../../ # 回到 omniserve 根目录
bash scripts/qserve_benchmark.sh
```
