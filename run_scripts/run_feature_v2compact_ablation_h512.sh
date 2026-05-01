#!/usr/bin/env bash
set -euo pipefail

# Run from the main WSL project folder:
# ~/lattice2026/lattice_area_paired_local_bundle/lattice_area_paired_local_bundle
#
# This runs the three h512 v2compact ablations sequentially:
# 1) neighbor_only
# 2) rank_only
# 3) v2_compact
#
# Logs are written to:
# outputs/feature_v2compact_ablation_logs/run_all_h512.log

if [ ! -d ".venv" ]; then
  echo "ERROR: .venv not found. Run this from the main project folder containing .venv, scripts, outputs, inputs."
  echo "Current directory: $(pwd)"
  exit 1
fi

if [ ! -f "scripts/train_force_curve_edgeset_fe_v2compact.py" ]; then
  echo "ERROR: scripts/train_force_curve_edgeset_fe_v2compact.py not found."
  echo "Copy the patched trainer into scripts/ first."
  exit 1
fi

ARTIFACT="outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
FAILURES="outputs/area_paired_v2_512x10/paired_failure_strains.npz"

if [ ! -f "$ARTIFACT" ]; then
  echo "ERROR: missing artifact zip:"
  echo "$ARTIFACT"
  exit 1
fi

if [ ! -f "$FAILURES" ]; then
  echo "ERROR: missing paired failure strains:"
  echo "$FAILURES"
  exit 1
fi

source .venv/bin/activate

mkdir -p outputs/feature_v2compact_ablation_logs

run_one () {
  local FEATURE_SET="$1"
  local OUT_DIR="$2"

  echo ""
  echo "============================================================"
  echo "Starting feature_set=${FEATURE_SET}"
  echo "Output: ${OUT_DIR}"
  echo "Time: $(date)"
  echo "============================================================"

  python scripts/train_force_curve_edgeset_fe_v2compact.py \
    --artifact-zip "$ARTIFACT" \
    --failure-strains "$FAILURES" \
    --out "$OUT_DIR" \
    --epochs 1000 \
    --patience 160 \
    --batch-size 96 \
    --hidden 512 \
    --lr 0.0012 \
    --energy-loss-weight 0.2 \
    --feature-set "$FEATURE_SET"

  echo ""
  echo "Finished feature_set=${FEATURE_SET}"
  echo "Time: $(date)"
}

(
  run_one neighbor_only outputs/edgeset_force_curve_fe_v2compact_neighboronly_h512
  run_one rank_only outputs/edgeset_force_curve_fe_v2compact_rankonly_h512
  run_one v2_compact outputs/edgeset_force_curve_fe_v2compact_combined_h512

  echo ""
  echo "============================================================"
  echo "ALL THREE RUNS COMPLETE"
  echo "Time: $(date)"
  echo "============================================================"

  echo ""
  echo "Metric files:"
  ls -lh \
    outputs/edgeset_force_curve_fe_v2compact_neighboronly_h512/metrics_summary.csv \
    outputs/edgeset_force_curve_fe_v2compact_rankonly_h512/metrics_summary.csv \
    outputs/edgeset_force_curve_fe_v2compact_combined_h512/metrics_summary.csv

) 2>&1 | tee outputs/feature_v2compact_ablation_logs/run_all_h512.log
