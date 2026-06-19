#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA40
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 1-00:00:00
#SBATCH -J sft_1.5b
#SBATCH -o sft_1.5b_%j.log

# Stage-1 SFT baseline = CODI teacher. 4 epochs, ckpt every 1000 steps (keep all).
# Cache from precompute.py (data/cache/sft_train); same filtered subset as CODI.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_sft \
  --model model_weights/qwen2.5-coder-1.5b \
  --output_dir model_weights/sft-coder-1.5b \
  --cache_dir data/cache/sft_train \
  --max_seq_len 3072 \
  --epochs 4 \
  --lr 2e-5 \
  --batch_size 2 \
  --grad_accum 4
