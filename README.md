# LATTICE Nest — Robust Ductility Experiments

Exploratory research code and compact result summaries for surrogate-guided robust ductility optimization in a fixed 3D nested-pyramid truss lattice.

## Current project state

Current best validated area design:

- `surrogate_opt_from_288`
- validated with 80 paired stochastic fracture simulations
- mean energy ≈ `2.0216`
- median energy ≈ `2.0615`
- p10 energy ≈ `1.7590`
- singular/local-mechanism count: `0 / 80`
- direct paired comparison vs original design `288`: wins `67 / 80`

This repository intentionally tracks lightweight code, notes, prompts, and compact CSV summaries. Large raw simulation artifacts, `.npz` arrays, model checkpoints, and generated ZIP files are excluded from Git and documented in `data_manifests/local_artifact_manifest.md`.

## Main ideas

- Fixed 3D nested-pyramid topology.
- Robust ductility objective: area under the force-displacement curve.
- Per-member area multipliers under a length-weighted volume constraint.
- Stochastic Weibull failure strain field.
- Fast EdgeSetMLP force-curve surrogate.
- Separate inference surrogate and gradient-ready optimization surrogate.
- Surrogate-gradient area optimization validated by direct simulation.

## Important guardrails

Do not use same-run post-fracture outcomes as surrogate inputs. Forbidden input features include force curve, energy, peak force, cascade size, terminal damage, and run status / solve failure flag.

Splits should be by `design_id`, not random row split.

## AI assistance note

This project code and analysis workflow were developed with substantial AI assistance. Simulation results and claims should be evaluated through the tracked validation artifacts and CSV summaries.
