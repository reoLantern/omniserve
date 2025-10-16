# Download the model config files for benchmarking
MODEL_CONFIG_DIR_PATH=./QServe-benchmarks

if [ ! -d "$MODEL_CONFIG_DIR_PATH" ]; then
    git clone https://www.modelscope.cn/datasets/mit-han-lab/QServe-benchmarks
fi

export CUDA_VISIBLE_DEVICES=1
export USE_RANDOM_TOKENS=1     # set to 1 to generate random tokens after layer to avoid bug

# Benchmark Llama-3-8B
MODEL=$MODEL_CONFIG_DIR_PATH/Llama-3-8B
# MODEL=/home/mingyuan.ma/work/omniserve/qserve_checkpoints/Llama-3-8B-QServe
GLOBAL_BATCH_SIZE=128 NUM_GPU_PAGE_BLOCKS=3200 \
python qserve_benchmark.py \
    --model $MODEL --benchmarking --precision w4a8kv4 --group-size -1 \
    # --quant-path /home/mingyuan.ma/work/omniserve/qserve_checkpoints/Llama-3-8B-QServe
