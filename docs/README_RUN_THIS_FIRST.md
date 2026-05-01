# LATTICE Area-Conditioned Paired Local Bundle

This bundle is meant to be copied to your WSL2 Ubuntu machine and run locally.

It contains:

- `inputs/lattice_fracture_nest_987654_robust_validation.zip`
  - repo/code with the nested lattice solver
- `inputs/nest_987654_shape15_s012_prune0_area_conditioned_artifacts.zip`
  - first area-conditioned pilot, including the 64 design definitions
- `inputs/nest_challenge_v1_candidate_shape15_s012_prune0_n1000_verified_package.zip`
  - verified fixed-area candidate reference package
- `inputs/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip`
  - partial paired-run artifact, included for reference
- `scripts/run_area_conditioned_paired_local.py`
  - local paired-seed runner
- `run_local_9designs.sh`
  - quick focused run: 9 designs x 16 paired seeds = 144 simulations
- `run_local_32designs.sh`
  - larger follow-up: 32 designs x 16 paired seeds = 512 simulations

## Fastest way to start

Copy this ZIP to WSL, then run:

    unzip lattice_area_paired_local_bundle.zip
    cd lattice_area_paired_local_bundle
    bash run_local_9designs.sh

That is the recommended first local run.

## If you have more time

After the 9-design run finishes, run:

    bash run_local_32designs.sh

## Outputs

The 9-design run writes to:

    outputs/area_paired_9designs/

The 32-design run writes to:

    outputs/area_paired_32designs/

Each output folder should include:

- run atomic `.npz` files
- status sidecars
- design table CSV
- run table CSV
- pairwise comparison CSV
- summary JSON
- report MD
- figures
- artifact ZIP

## Parallel workers

Both shell scripts default to `--workers 4`.

If your machine is comfortable, edit the shell script and increase to `--workers 6` or `--workers 8`.

If it gets too hot or unstable, reduce to `--workers 2`.

## What this run does

It evaluates selected area designs using common paired Weibull failure-strain fields:

    failure_seed = 3200000 + sample_id

This means every design is tested against the same 16 random failure fields, so ratios vs the uniform baseline are much more meaningful.

## Important

This remains experimental robust-design work.

Do not treat the result as official.
Do not treat it as full robust optimization.
Do not mix it into the fixed-area candidate dataset.
