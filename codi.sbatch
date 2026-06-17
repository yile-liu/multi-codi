#!/bin/bash
#SBATCH -A berzelius-2026-167
#SBATCH -C thin
#SBATCH --gpus=4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_trace
#SBATCH -o codi_trace_%j.log

# Stage 2b: per-frame (multi-span) CODI self-distillation. teacher=student init from Stage-1 SFT.
# 4x A100-40GB DDP. Cache built offline on the login node (precompute.py, CPU-only).
export PATH=/proj/assert-berzelius/users/x_sirli/conda/envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) train_codi.py \
  --model model_weights/sft-coder-1.5b \
  --output_dir model_weights/codi-coder-1.5b \
  --cache_dir data/cache/codi_train \
  --latent_steps 1 \
  --epochs 10 \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum 4
