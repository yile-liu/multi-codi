#!/bin/bash
#SBATCH -A berzelius-2026-167
#SBATCH -C thin
#SBATCH --gpus=1
#SBATCH -t 04:00:00
#SBATCH -J sft_trace
#SBATCH -o sft_trace_%j.log

# Stage 1: full explicit-trace SFT = baseline = teacher. Qwen2.5-Coder-1.5B, full FT.
# thin node = A100-40GB: gradient checkpointing on, bs2*ga8 (eff 16).
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_sft.py \
  --model model_weights/qwen2.5-coder-1.5b \
  --output_dir model_weights/sft-coder-1.5b \
  --split train \
  --max_seq_len 4096 \
  --epochs 3 \
  --lr 1e-5 \
  --batch_size 2 \
  --grad_accum 8
