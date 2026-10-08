# LATTICE Nest — Robust Ductility Experiments

Exploratory research code and compact result summaries for surrogate-guided robust ductility optimization in a fixed 3D nested-pyramid truss lattice.

## Start here

This is a research code and results repository, with no packaged interactive demo.
Clone it to inspect the committed summaries:

```sh
git clone https://github.com/CharlesMish/lattice_nest.git
cd lattice_nest
```

- [Compact results](results_small/README.md) and the
  [paired comparison with design 288](results_small/surrogate_gradient_opt_top5_move025_risk035_validation_80seeds/compare_opt001_vs_source288.csv)
  provide a data-only starting point; no simulation environment is needed to read the CSVs.
- [Current status](docs/CURRENT_STATUS.md) records surrogate roles and proposed experiments.
- [Local artifact manifest](data_manifests/local_artifact_manifest.md) lists the
  untracked training arrays, simulation outputs and model checkpoint.

The full solver/input bundles and trained checkpoints are **not included in this
repository**. [Run-this-first](docs/README_RUN_THIS_FIRST.md) describes a separate
WSL bundle, including ZIP inputs and launch scripts absent from a fresh clone.
It is historical bundle setup, not a standalone quickstart for this checkout.
Likewise, the [feature-builder report](docs/feature_v2compact_builder_report.md)
records what was available in its original execution environment; its sandbox
paths and unrun checks remain provenance, not instructions for your machine.
Full simulation or training requires the corresponding external artifacts and
an explicitly prepared environment; the compact CSVs alone do not reproduce it.

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

## Contributing

Propose a focused change with its input provenance, design-level split, paired
seed policy and validation limits. Keep raw artifacts out of Git and update the
manifest when documenting a new external dataset. Distinguish a synthetic smoke
test from real-data training and a direct simulation from surrogate prediction.
The leakage and split guardrails above apply to every new experiment.

## License

See the existing [LICENSE](LICENSE) for the repository's terms.
