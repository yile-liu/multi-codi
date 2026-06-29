#!/bin/bash
#SBATCH -A naiss2026-3-261
#SBATCH --gpus-per-node=T4:4
#SBATCH -t 0-01:00:00
#SBATCH -J precompute
#SBATCH -o logs/precompute_%j.log

# CPU-bound trace/token cache build (no GPU math; T4 slice just for the node + cores).
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0

python precompute.py \
  --model "${MODEL:-model_weights/qwen2.5-coder-1.5b}" \
  --mode codi \
  --sources mbpp humaneval pyx \
  --max_seq_len 6144 --max_frames 256 \
  --workers "${WORKERS:-16}" \
  --out "${OUT:-data/cache/codi_train_delta}" --overwrite
