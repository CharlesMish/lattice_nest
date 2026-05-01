# LATTICE Feature v2 Compact Patch — Builder Report

## Status

Implemented `train_force_curve_edgeset_fe_v2compact.py` from the uploaded `train_force_curve_edgeset_fe.py`.

The real `outputs/area_paired_v2_512x10/...` training artifacts are **not present in this sandbox**, so the h512 real ablation matrix could not be executed here. I verified this by searching available `/mnt/data` files. The script was compiled and smoke-tested on a synthetic area-conditioned fixture with the same required archive member suffixes.

## Implemented feature sets

| feature_set | total edge dim | base dim | added dim | finite check |
|---|---:|---:|---:|---|
| base | 38 | 38 | 0 | pass |
| neighbor_only | 47 | 38 | 9 | pass |
| rank_only | 42 | 38 | 4 | pass |
| v2_compact | 51 | 38 | 13 | pass |

## Added feature names

Neighbor features:
- `neighbor_mean_area_mult`
- `neighbor_std_area_mult`
- `neighbor_mean_elastic_demand`
- `neighbor_max_elastic_demand`
- `neighbor_mean_failure_margin`
- `neighbor_min_failure_margin`
- `neighbor_mean_demand_ratio`
- `neighbor_max_demand_ratio`
- `neighbor_degree`

Global rank/percentile features:
- `area_global_pct`
- `demand_global_pct`
- `margin_global_pct`
- `demand_ratio_global_pct`

## Leakage audit

- No mechanics, solver, simulation, dataset rows, Weibull law, area constraints, or run-status handling were modified.
- No target/post-run features were added as model inputs.
- `force_curve`, `energy`, `peak_force`, `final/peak`, cascade metrics, terminal damage, solve status, runtime, and same-run fracture diagnostics are not used as edge inputs.
- `run_status` and `force_valid_mask` remain only for filtering/masking/evaluation.
- Rank features are computed independently within each kept run/design graph over its members, not from dataset-wide train/val/test thresholds.
- Split policy remains the original design-level split via `make_design_splits`.

## Synthetic smoke test

Command used equivalent to:

```bash
python train_force_curve_edgeset_fe_v2compact.py \
  --artifact-zip synthetic_area_conditioned_paired_artifacts.zip \
  --failure-strains paired_failure_strains.npz \
  --out smoke_v2_compact \
  --epochs 1 \
  --patience 1 \
  --batch-size 32 \
  --hidden 64 \
  --lr 0.001 \
  --energy-loss-weight 0.2 \
  --feature-set v2_compact \
  --device cpu
```

Result:
- compile: pass
- old/base feature dim: 38
- v2_compact feature dim: 51
- finite check: pass
- one epoch: started and completed
- output: `/mnt/data/feature_v2compact_synthetic/smoke_v2_compact`

The synthetic metrics are not scientifically meaningful; they only verify execution and artifact writing.

## Available real metrics from uploaded CSVs

Baseline h512 full-feature test:
- force RMSE: 1.583929
- force R²: 0.980733
- mean per-step R²: 0.964525
- energy RMSE: 0.030119
- energy R²: 0.987731

No-elastic h512 test:
- force RMSE: 2.426438
- force R²: 0.954784
- mean per-step R²: 0.909402
- energy RMSE: 0.068956
- energy R²: 0.935692

Interpretation: no-elastic remains a strong negative ablation. Elastic-demand/failure-margin features are essential and should remain in the base feature set.

## Real ablation matrix status

| variant | real h512 metrics |
|---|---|
| baseline | read from uploaded CSV |
| no-elastic | read from uploaded CSV |
| neighbor_only | not run: real v2 512x10 artifacts absent in sandbox |
| rank_only | not run: real v2 512x10 artifacts absent in sandbox |
| v2_compact | not run: real v2 512x10 artifacts absent in sandbox |

## Recommended next command on WSL

From the expected project root, copy `train_force_curve_edgeset_fe_v2compact.py` to `scripts/`, then run:

```bash
python scripts/train_force_curve_edgeset_fe_v2compact.py \
  --artifact-zip outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip \
  --failure-strains outputs/area_paired_v2_512x10/paired_failure_strains.npz \
  --out outputs/feature_v2compact_smoke \
  --epochs 1 \
  --patience 1 \
  --batch-size 32 \
  --hidden 64 \
  --lr 0.001 \
  --energy-loss-weight 0.2 \
  --feature-set v2_compact
```

Then run the h512 ablation matrix exactly as requested for `neighbor_only`, `rank_only`, and `v2_compact`.

## Recommendation

Do not adopt the compact features yet on synthetic smoke alone. The implementation is ready for real h512 testing. Keep the old full-feature h512 baseline as the benchmark to beat. If `neighbor_only`, `rank_only`, or `v2_compact` matches or beats the baseline on the default split, then run the split-stability gate on seeds `20260431`, `20260432`, and `20260433`.
