#!/bin/bash
# Rename model_weights CODI/SFT dirs to match results naming (file-stem style).
# Safe-guard: refuses to run while any siruil job is still queued/running,
# since renaming an in-use dir splits training outputs / breaks eval MODEL paths.
set -euo pipefail
cd "$(dirname "$0")/model_weights"

if squeue -u siruil -h -o "%i" | grep -q .; then
  echo "ABORT: jobs still in queue. Wait until 'squeue -u siruil' is empty."
  squeue -u siruil -o "%.10i %.20j %.2t"
  exit 1
fi

# current -> new
declare -A MAP=(
  [codi-multi-1.5b]=codi1.5b_a1.0_b1.0_g1.0_ls1
  [codi-multi-3b]=codi3b_a1.0_b1.0_g1.0_ls1
  [codi-multi-1.5b-wt]=codi1.5b_a0.5_b1.0_g0.5_ls1
  [codi-multi-1.5b-ls2]=codi1.5b_a1.0_b1.0_g1.0_ls2
  [codi-multi-1.5b-wt-ls2]=codi1.5b_a0.5_b1.0_g0.5_ls2
  [codi-multi-3b-wt]=codi3b_a0.5_b1.0_g0.5_ls1
  [codi-multi-3b-ls2]=codi3b_a1.0_b1.0_g1.0_ls2
  [codi-multi-3b-wt-ls2]=codi3b_a0.5_b1.0_g0.5_ls2
  [sft-coder-1.5b]=sft1.5b_lr1e5_bs32
  [sft-coder-3b]=sft3b_lr1e5_bs32
  [sft-coder-3b-lr1e5-bs64]=sft3b_lr1e5_bs64
)

for old in "${!MAP[@]}"; do
  new="${MAP[$old]}"
  if [ -e "$new" ]; then echo "skip $old -> $new (target exists)"; continue; fi
  if [ -d "$old" ]; then mv -v "$old" "$new"; else echo "skip $old (missing)"; fi
done
echo "done."
