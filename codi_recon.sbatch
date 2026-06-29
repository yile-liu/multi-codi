#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH -C 4xA100fat
#SBATCH --gpus-per-node=A100fat:4
#SBATCH -t 1-00:00:00
#SBATCH -J codi_recon
#SBATCH -o logs/codi_recon_%j.log

# Co-trained shared-weight CODI (like codi_multi) + locals reconstruction (method 1).
# Forces the latent block to encode the dropped $LOCALS via a reconstruction CE on lm_head.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MODEL/OUTPUT_DIR/CACHE_DIR/MAX_STEPS/LATENT_STEPS/SAVE_STEPS/ALPHA/BETA/GAMMA/RECON_W/MAX_RECON_LEN/DEBUG_RECON_PRINT overridable via --export.
torchrun --nproc_per_node=4 --master_port=$((20000 + RANDOM % 10000)) -m train.train_codi_recon \
  --model "${MODEL:-model_weights/temp}" \
  --output_dir "${OUTPUT_DIR:-model_weights/codi-recon-temp}" \
  --cache_dir "${CACHE_DIR:-data/cache/codi_train_delta}" \
  --max_seq_len 3072 \
  --latent_steps "${LATENT_STEPS:-1}" \
  --epochs 5 \
  --max_steps "${MAX_STEPS:-1500}" \
  --save_steps "${SAVE_STEPS:-300}" \
  --alpha "${ALPHA:-1.0}" \
  --beta "${BETA:-1.0}" \
  --gamma "${GAMMA:-1.0}" \
  --recon_w "${RECON_W:-0.1}" \
  --max_recon_len "${MAX_RECON_LEN:-64}" \
  --debug_recon_print "${DEBUG_RECON_PRINT:-0}" \
  --lr 1e-5 \
  --batch_size "${BATCH:-1}" \
  --grad_accum "${ACCUM:-4}"
