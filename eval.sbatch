#!/bin/bash
#SBATCH -A naiss2026-3-261
# -C selects node type (nodes with 4×A100-40GB); --gpus-per-node allocates how many this job uses.
#SBATCH -C 4xA100
#SBATCH --gpus-per-node=A100:2
#SBATCH -t 02:00:00
#SBATCH -J eval_crux
#SBATCH -o eval_crux_%j.log

# Pre-requisite: CRUXEVAL_DIR must point to a save_to_disk copy of the test split.
# MODEL/N/OUT overridable via --export when submitting with sbatch.
export PATH=/mimer/NOBACKUP/groups/assert-kth/siruil/conda_envs/CWM/bin:$PATH
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONHASHSEED=0

torchrun --nproc_per_node=2 --master_port=$((20000 + RANDOM % 10000)) -m "${SCRIPT:-eval.eval_cruxeval_sft}" \
  --model "${MODEL:-model_weights/sft-coder-1.5b/checkpoint-6928}" \
  --n_samples "${N:--1}" \
  --out "${OUT:-results/cruxeval_ckpt6928.json}"
