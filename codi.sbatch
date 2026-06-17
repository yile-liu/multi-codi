#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA100
#SBATCH --gpus-per-node=A100:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_trace
#SBATCH -o codi_trace_%j.log

# Full-layer KD (kd_layers=None default) + multi-anchor (single_anchor=False default).
# 4x A100-40GB DDP. Cache from precompute.py (cache_dir default); paged_adamw_8bit + teacher grad-ckpt.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) train_codi.py \
  --model model_weights/sft-coder-1.5b/checkpoint-10392 \
  --output_dir model_weights/codi-coder-1.5b-ckpt10392 \
  --max_seq_len 3072 \
  --latent_steps 1 \
  --epochs 5 \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum 4
