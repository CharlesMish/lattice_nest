#!/usr/bin/env bash
set -euo pipefail

# Faster version: combine data, then train only opt_area_context.
source .venv/bin/activate

python scripts/make_area_paired_fullcombo.py

python scripts/train_force_curve_edgeset_optfeatures.py \
  --artifact-zip outputs/area_paired_v2_fullcombo/nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip \
  --failure-strains outputs/area_paired_v2_fullcombo/paired_failure_strains.npz \
  --out outputs/edgeset_force_curve_opt_area_context_fullcombo_h512 \
  --feature-set opt_area_context \
  --epochs 1000 \
  --patience 160 \
  --batch-size 96 \
  --hidden 512 \
  --lr 0.0012 \
  --energy-loss-weight 0.2
