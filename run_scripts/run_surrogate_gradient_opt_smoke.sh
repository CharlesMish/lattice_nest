#!/usr/bin/env bash
set -euo pipefail

# Faster smoke optimization from only top design 298.
source .venv/bin/activate

python scripts/optimize_area_with_surrogate.py \
  --model outputs/edgeset_force_curve_opt_area_context_fullcombo_h512/model.pt \
  --design-artifact outputs/area_paired_v2_fullcombo/nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip \
  --failure-strains outputs/area_paired_v2_fullcombo/paired_failure_strains.npz \
  --out outputs/surrogate_gradient_opt_smoke_298 \
  --start-design-ids 298 \
  --n-failure-samples 16 \
  --steps 200 \
  --lr 0.035 \
  --risk-weight 0.15 \
  --volume-penalty 1000.0 \
  --move-penalty 0.0
