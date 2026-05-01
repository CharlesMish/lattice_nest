#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
ARTIFACT="outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
FAILURES="outputs/area_paired_v2_512x10/paired_failure_strains.npz"
mkdir -p outputs/optfeature_ablation_logs
run_one () {
  local FEATURE_SET="$1"
  local OUT_DIR="$2"
  echo ""
  echo "============================================================"
  echo "Starting opt feature set: $FEATURE_SET"
  echo "Output: $OUT_DIR"
  echo "Time: $(date)"
  echo "============================================================"
  python scripts/train_force_curve_edgeset_optfeatures.py \
    --artifact-zip "$ARTIFACT" \
    --failure-strains "$FAILURES" \
    --out "$OUT_DIR" \
    --feature-set "$FEATURE_SET" \
    --epochs 1000 \
    --patience 160 \
    --batch-size 96 \
    --hidden 512 \
    --lr 0.0012 \
    --energy-loss-weight 0.2
  echo "Finished $FEATURE_SET at $(date)"
}
(
  run_one opt_basic outputs/edgeset_force_curve_opt_basic_h512
  run_one opt_area_context outputs/edgeset_force_curve_opt_area_context_h512
  echo ""
  echo "DONE. Metrics:"
  ls -lh \
    outputs/edgeset_force_curve_opt_basic_h512/metrics_summary.csv \
    outputs/edgeset_force_curve_opt_area_context_h512/metrics_summary.csv
) 2>&1 | tee outputs/optfeature_ablation_logs/run_optfeatures_h512.log
