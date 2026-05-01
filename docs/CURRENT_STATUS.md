# Current Status

## Best validated design

`surrogate_opt_from_288`

80-seed validation:

- mean energy: `2.0216`
- median energy: `2.0615`
- p10 energy: `1.7590`
- singular/local-mechanism: `0 / 80`

Direct comparison vs original `288`:

- mean ratio: about `1.0338x`
- median ratio: about `1.0230x`
- beats source: `67 / 80`

## Best surrogate roles

### Best evaluator / inference surrogate

`h512 rank_only EdgeSetMLP`

Adds within-run percentile/rank features to full feature-engineered EdgeSetMLP.

### Best gradient-ready optimizer surrogate

`opt_area_context_fullcombo_h512`

Uses static geometry + direct area features + differentiable area-context features. No hard ranks, no elastic solve, no post-run leakage.

## Next recommended experiments

1. 160–256 seed validation comparing original 288, original 298, surrogate_opt_from_288, and surrogate_opt_from_298.
2. Area-cap stress test with max area around 2.7–2.85.
3. Rank-only evaluator audit of optimized candidates.
4. Small corotational smoke test.
