#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA100fat
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_trace
#SBATCH -o codi_trace_%j.log

# Full-layer KD (kd_layers=None default) + multi-anchor (single_anchor=False default).
# 4x A100fat (80GB) DDP. Cache from precompute.py (cache_dir default); adamw_bnb_8bit + teacher grad-ckpt.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL/OUTPUT_DIR/MAX_STEPS/LATENT_STEPS/SAVE_STEPS/ALPHA/BETA/GAMMA overridable via --export.
torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_codi \
  --model "${MODEL:-model_weights/temp}" \
  --output_dir "${OUTPUT_DIR:-model_weights/codi-temp}" \
  --max_seq_len 3072 \
  --latent_steps "${LATENT_STEPS:-1}" \
  --epochs 5 \
  --max_steps "${MAX_STEPS:-3000}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --alpha "${ALPHA:-1.0}" \
  --beta "${BETA:-1.0}" \
  --gamma "${GAMMA:-1.0}" \
  --lr 1e-5 \
  --batch_size "${BATCH:-1}" \
  --grad_accum "${ACCUM:-4}"
