#!/usr/bin/env bash
set -euo pipefail

source .venv/bin/activate
python scripts/make_pooling_trainer.py

BASE_OUT="outputs/pooling_ablation_fe_v2_512x10_h256"
mkdir -p "$BASE_OUT"

for POOL in mean_sum mean_std mean_std_max mean_std_max_min; do
  echo ""
  echo "=============================="
  echo "Pooling ablation h=256: $POOL"
  echo "=============================="
  python scripts/train_force_curve_edgeset_fe_pooling.py \
    --artifact-zip outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip \
    --failure-strains outputs/area_paired_v2_512x10/paired_failure_strains.npz \
    --out "$BASE_OUT/$POOL" \
    --epochs 700 \
    --patience 100 \
    --batch-size 128 \
    --hidden 256 \
    --lr 0.0015 \
    --energy-loss-weight 0.2 \
    --pooling "$POOL"
done

python scripts/summarize_pooling_ablation.py --base-out "$BASE_OUT"
