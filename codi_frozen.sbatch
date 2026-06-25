#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA100fat
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_frozen
#SBATCH -o codi_frozen_%j.log

# Frozen SFT teacher (no teacher CE) + last-layer "key hidden" KD. KD_TARGET=hidden|logit.
# Student and teacher both init from the SFT checkpoint; eliminates the co-trained-teacher degradation variable.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL=student init (=SFT ckpt); FROZEN_TEACHER defaults to MODEL. KD_TARGET/LATENT_STEPS/OUTPUT_DIR overridable.
torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_codi \
  --model "${MODEL:?set MODEL to the SFT checkpoint}" \
  --frozen_teacher "${FROZEN_TEACHER:-${MODEL}}" \
  --output_dir "${OUTPUT_DIR:?set OUTPUT_DIR}" \
  --kd_target "${KD_TARGET:-hidden}" \
  --kd_temp "${KD_TEMP:-2.0}" \
  --alpha 0 --beta 1.0 --gamma "${GAMMA:-1.0}" \
  --max_seq_len 3072 \
  --latent_steps "${LATENT_STEPS:-1}" \
  --epochs 5 \
  --max_steps "${MAX_STEPS:-3000}" \
  --save_steps "${SAVE_STEPS:-500}" \
  --lr 1e-5 \
  --batch_size "${BATCH:-1}" \
  --grad_accum "${ACCUM:-4}"
