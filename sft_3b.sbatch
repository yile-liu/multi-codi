#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA100
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 1-00:00:00
#SBATCH -J sft_qwen2.5_coder_3b
#SBATCH -o sft_qwen2.5_coder_3b_%j.log

export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Overridable from the command line, e.g.:
#   sbatch --export=ALL,LR=1e-5,GRAD_ACCUM=16,OUTPUT_DIR=model_weights/sft-coder-3b-lr1e5-bs64 sft_3b.sbatch
LR=${LR:-2e-5}
GRAD_ACCUM=${GRAD_ACCUM:-8}
OUTPUT_DIR=${OUTPUT_DIR:-model_weights/sft-coder-3b}

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_sft \
  --model model_weights/qwen2.5-coder-3b \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir data/cache/sft_train \
  --max_seq_len 3072 \
  --epochs 4 \
  --lr "$LR" \
  --batch_size 1 \
  --grad_accum "$GRAD_ACCUM"
