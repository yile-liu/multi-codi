#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA40
#SBATCH --gpus-per-node=A40:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_single
#SBATCH -o codi_single_%j.log

# Faithful single-block CODI (arXiv 2502.21074): one latent block replaces the whole
# trace, student emits only the answer, KD at a single anchor (L1/std, all layers).
# Separate pipeline from per-line codi_multi. 4x A40 (44GB) DDP.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL/OUTPUT_DIR/MAX_STEPS/LS/SAVE_STEPS/ALPHA/BETA/GAMMA/BATCH/ACCUM overridable via --export.
torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_codi_single \
  --model "${MODEL:-model_weights/sft-coder-1.5b/checkpoint-6936}" \
  --output_dir "${OUTPUT_DIR:-model_weights/codi-single-1.5b}" \
  --max_seq_len 3072 \
  --latent_steps "${LS:-6}" \
  --epochs 5 \
  --max_steps "${MAX_STEPS:-1000}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --alpha "${ALPHA:-1.0}" \
  --beta "${BETA:-1.0}" \
  --gamma "${GAMMA:-1.0}" \
  --lr 1e-5 \
  --batch_size "${BATCH:-1}" \
  --grad_accum "${ACCUM:-4}"
