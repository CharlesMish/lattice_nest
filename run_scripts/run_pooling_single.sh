#!/usr/bin/env bash
set -euo pipefail

POOL="${1:-mean_std_max}"
HIDDEN="${2:-512}"

source .venv/bin/activate
python scripts/make_pooling_trainer.py

OUT="outputs/edgeset_force_curve_fe_v2_512x10_h${HIDDEN}_${POOL}"

if [ "$HIDDEN" -ge 512 ]; then
  BATCH=96
  LR=0.0012
  EPOCHS=1000
  PATIENCE=160
else
  BATCH=128
  LR=0.0015
  EPOCHS=700
  PATIENCE=100
fi

python scripts/train_force_curve_edgeset_fe_pooling.py \
  --artifact-zip outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip \
  --failure-strains outputs/area_paired_v2_512x10/paired_failure_strains.npz \
  --out "$OUT" \
  --epochs "$EPOCHS" \
  --patience "$PATIENCE" \
  --batch-size "$BATCH" \
  --hidden "$HIDDEN" \
  --lr "$LR" \
  --energy-loss-weight 0.2 \
  --pooling "$POOL"
