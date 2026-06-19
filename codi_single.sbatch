#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA40
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_single
#SBATCH -o codi_single_%j.log

# Single-anchor CODI ablation (vanilla CODI): KD at the last frame only, everything
# else identical to the multi-anchor codi-coder-1.5b-ckpt10392 run (same SFT ckpt,
# 3072 seq / 1000 steps / save 500 / lr 1e-5 / bs1*accum4, kd_layers=None=all layers)
# so single-vs-multi is a clean comparison. Fresh run on 4x A40 (44GB); fits at 3072
# (the prior A40 OOMs were resume-only, the "A40 resume bug").
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_codi \
  --model model_weights/sft-coder-1.5b/checkpoint-10392 \
  --output_dir model_weights/codi-single-coder-1.5b-ckpt10392 \
  --max_seq_len 3072 \
  --latent_steps 1 \
  --epochs 5 \
  --max_steps 1000 \
  --save_steps 500 \
  --lr 1e-5 \
  --batch_size 1 \
  --grad_accum 4 \
  --single_anchor
