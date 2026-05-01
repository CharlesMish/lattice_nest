#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python scripts/train_force_curve_edgeset_optfeatures.py \
  --artifact-zip outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip \
  --failure-strains outputs/area_paired_v2_512x10/paired_failure_strains.npz \
  --out outputs/optfeature_smoke_basic \
  --feature-set opt_basic \
  --epochs 1 --patience 1 --batch-size 32 --hidden 64 --lr 0.001 --energy-loss-weight 0.2
python scripts/train_force_curve_edgeset_optfeatures.py \
  --artifact-zip outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip \
  --failure-strains outputs/area_paired_v2_512x10/paired_failure_strains.npz \
  --out outputs/optfeature_smoke_area_context \
  --feature-set opt_area_context \
  --epochs 1 --patience 1 --batch-size 32 --hidden 64 --lr 0.001 --energy-loss-weight 0.2
