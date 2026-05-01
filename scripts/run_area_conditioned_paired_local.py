#!/usr/bin/env python3
"""
Local paired-seed area-conditioned robust ductility runner.

This script reuses area multipliers from the first area-conditioned pilot,
then evaluates selected designs on common Weibull failure-strain fields.

Default quick run:
  design IDs 000,001,002,023,026,029,046,047,061
  sample IDs 0..15
  failure_seed = 3200000 + sample_id

Outputs:
  - atomic per-run NPZ files
  - status sidecars
  - design/run/pairwise CSVs
  - summary JSON
  - report MD
  - figures
  - final artifact ZIP
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
import zipfile
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lattice_fracture.batch import _summarize_run
from lattice_fracture.distributions import sample_failure_strains
from lattice_fracture.failure import ProgressiveFailure
from lattice_fracture.nested_tuning_recon import NestedGeometrySetting, _candidate_for_geometry
from lattice_fracture.tapered_failure_recon import _quality_gated_lattice


GEOMETRY_NAME = "nest_987654_s075"
DENSITIES = (9, 8, 7, 6, 5, 4)
LAYER_SPACING = 0.75
SHAPE = 1.5
SCALE = 0.012
INITIAL_PRUNE_FRAC = 0.0
MAX_DISP = 0.05
N_STEPS = 100
MAX_CASCADE_ITER = 100
E = 1000.0
A_BASE = 1.0


def jsonable(x: Any):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if np.isnan(x):
            return None
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=jsonable), encoding="utf-8")


def parse_design_ids(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def load_pilot_designs(artifact_zip: Path):
    """Load area multipliers and design table from the first pilot artifact ZIP."""
    with zipfile.ZipFile(artifact_zip) as z:
        setup_name = next(n for n in z.namelist() if n.endswith("area_pilot_setup.npz"))
        design_table_name = next(
            n for n in z.namelist()
            if n.endswith("area_conditioned_design_table.csv")
            or n.endswith("_design_table.csv")
        )
        setup = np.load(io.BytesIO(z.read(setup_name)), allow_pickle=True)
        design_table = pd.read_csv(io.BytesIO(z.read(design_table_name)))

        data = {
            "nodes": np.asarray(setup["nodes"], dtype=np.float64),
            "members": np.asarray(setup["members"], dtype=np.int64),
            "member_lengths": np.asarray(setup["member_lengths"], dtype=np.float64),
            "member_layer": np.asarray(setup["member_layer"], dtype=np.int16),
            "family_code": np.asarray(setup["family_code"], dtype=np.int16),
            "family_names": [str(x) for x in setup["family_names"].tolist()],
            "boundary_member_mask": np.asarray(setup["boundary_member_mask"], dtype=bool),
            "loaded_support_mask": np.asarray(setup["loaded_support_mask"], dtype=bool),
            "bottom_touch_mask": np.asarray(setup["bottom_touch_mask"], dtype=bool),
            "area_multipliers": np.asarray(setup["area_multipliers"], dtype=np.float64),
            "displacement_schedule": np.asarray(setup["displacement_schedule"], dtype=np.float64),
            "design_table": design_table,
        }
    return data


def make_lattice(seed: int):
    geom = NestedGeometrySetting(
        GEOMETRY_NAME,
        DENSITIES,
        layer_spacing=LAYER_SPACING,
        initial_prune_frac=INITIAL_PRUNE_FRAC,
    )
    candidate = _candidate_for_geometry(geom)
    lattice, gate_info = _quality_gated_lattice(
        candidate,
        base_seed=int(seed),
        E=E,
        A=A_BASE,
        probe_disp=max(MAX_DISP / max(N_STEPS, 1), 1e-6),
    )
    return lattice, gate_info


def status_from_row(row: dict, forces: np.ndarray) -> tuple[str, int, int, np.ndarray, bool]:
    """Return run_status, terminal_step, curve_valid_until_step, force_valid_mask, final_force_is_physical."""
    n = len(forces)
    mask = np.zeros(N_STEPS, dtype=bool)
    raw_n = min(n, N_STEPS)

    if int(row.get("n_solve_failures", 0)) > 0:
        # ProgressiveFailure records a zero force at the failed solve.
        valid_n = max(0, raw_n - 1)
        mask[:valid_n] = True
        return "singular_local_mechanism", raw_n - 1, valid_n - 1, mask, False

    if raw_n < N_STEPS:
        valid_n = raw_n
        mask[:valid_n] = True
        return "truncated", raw_n - 1, valid_n - 1, mask, False

    mask[:raw_n] = True
    peak = float(np.max(np.abs(forces[:raw_n]))) if raw_n else 0.0
    if peak <= 1e-12:
        return "near_zero", -1, raw_n - 1, mask, True
    return "complete_physical_curve", -1, raw_n - 1, mask, True


def run_one_task(task: dict) -> dict:
    out_path = Path(task["out_path"])
    status_path = Path(task["status_path"])
    failed_path = Path(task["failed_path"])
    tmp_path = out_path.with_suffix(".tmp.npz")

    if out_path.exists():
        try:
            with np.load(out_path, allow_pickle=True) as d:
                scalar = json.loads(str(np.asarray(d["scalar"]).item()))
            return {"ok": True, "skipped": True, "path": str(out_path), "design_id": task["design_id"], "sample_id": task["sample_id"], "status": scalar.get("run_status", "unknown")}
        except Exception:
            out_path.unlink(missing_ok=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    failed_path.parent.mkdir(parents=True, exist_ok=True)

    design_id = int(task["design_id"])
    sample_id = int(task["sample_id"])
    failure_seed = int(task["failure_seed"])
    area_multipliers = np.asarray(task["area_multipliers"], dtype=np.float64)
    member_lengths = np.asarray(task["member_lengths"], dtype=np.float64)
    design_meta = task["design_meta"]

    try:
        t0 = time.time()
        lattice, gate_info = make_lattice(failure_seed)
        if lattice.n_members != area_multipliers.shape[0]:
            raise RuntimeError(f"n_members mismatch: lattice={lattice.n_members}, area={area_multipliers.shape[0]}")

        failure_strains = sample_failure_strains(
            lattice.n_members,
            "weibull",
            seed=failure_seed,
            scale=SCALE,
            shape=SHAPE,
        )

        sim = ProgressiveFailure(lattice, E=E, A=area_multipliers, linear=True)
        result = sim.run(
            failure_strains,
            max_disp=MAX_DISP,
            n_steps=N_STEPS,
            max_cascade_iter=MAX_CASCADE_ITER,
            compression_factor=1.0,
        )
        row = _summarize_run(result, run_id=design_id * 1000 + sample_id, seed=failure_seed)
        row.update(gate_info)

        forces = np.asarray(row["forces"], dtype=np.float64)
        disps = np.asarray(row["displacements"], dtype=np.float64)
        run_status, terminal_step, curve_valid_until_step, force_valid_mask, final_force_is_physical = status_from_row(row, forces)

        valid_n = int(np.sum(force_valid_mask))
        if valid_n > 1:
            physical_forces = forces[:valid_n]
            physical_disps = disps[:valid_n]
            peak_force = float(np.max(np.abs(physical_forces)))
            energy = float(np.trapezoid(np.abs(physical_forces), physical_disps))
            final_peak = float(physical_forces[-1] / peak_force) if peak_force > 1e-15 else math.nan
        elif valid_n == 1:
            peak_force = float(np.abs(forces[0]))
            energy = 0.0
            final_peak = 1.0 if peak_force > 1e-15 else math.nan
        else:
            peak_force = 0.0
            energy = 0.0
            final_peak = math.nan

        failed_mask = np.asarray(row["member_fail_step"]) >= 0
        n_failures = int(np.sum(failed_mask))
        terminal_failed_frac = float(n_failures / float(lattice.n_members))

        total_volume = float(np.sum(area_multipliers * member_lengths))
        baseline_volume = float(np.sum(member_lengths))
        rel_volume_error = float((total_volume - baseline_volume) / baseline_volume)

        scalar = {
            "design_id": design_id,
            "design_name": design_meta.get("design_name", f"design_{design_id:03d}"),
            "design_category": design_meta.get("category", "unknown"),
            "sample_id": sample_id,
            "failure_seed": failure_seed,
            "run_status": run_status,
            "terminal_reason": run_status if run_status != "complete_physical_curve" else "none",
            "terminal_step": int(terminal_step),
            "curve_valid_until_step": int(curve_valid_until_step),
            "final_force_is_physical": bool(final_force_is_physical),
            "near_zero_flag": bool(run_status == "near_zero"),
            "peak_force": peak_force,
            "energy": energy,
            "final_peak": final_peak,
            "n_failures": n_failures,
            "n_tension_failures": int(row["n_tension_failures"]),
            "n_compression_failures": int(row["n_compression_failures"]),
            "max_cascade_size": int(row["max_cascade_size"]),
            "n_solve_failures": int(row["n_solve_failures"]),
            "terminal_failed_member_fraction": terminal_failed_frac,
            "total_volume": total_volume,
            "baseline_volume": baseline_volume,
            "relative_volume_error": rel_volume_error,
            "runtime_s": float(time.time() - t0),
        }

        if tmp_path.exists():
            tmp_path.unlink()
        np.savez_compressed(
            tmp_path,
            scalar=np.array(json.dumps(scalar, sort_keys=True)),
            failure_strains=np.asarray(row["failure_strains"], dtype=np.float64),
            displacements=disps,
            forces=forces,
            force_valid_mask=force_valid_mask,
            n_active=np.asarray(row["n_active"], dtype=np.int64),
            member_fail_step=np.asarray(row["member_fail_step"], dtype=np.int64),
            initial_active_mask=np.asarray(row["initial_active_mask"], dtype=bool),
            initial_damage_mask=np.asarray(row["initial_damage_mask"], dtype=bool),
            area_multipliers=area_multipliers.astype(np.float64),
            volume_contribution=(area_multipliers * member_lengths).astype(np.float64),
        )
        os.replace(tmp_path, out_path)

        status_record = {
            "scalar": scalar,
            "solve_statuses": result.get("solve_statuses", []),
            "failure_events": result.get("failure_events", []),
        }
        dump_json(status_path, status_record)
        failed_path.unlink(missing_ok=True)
        return {"ok": True, "skipped": False, "path": str(out_path), "design_id": design_id, "sample_id": sample_id, "status": run_status}

    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        info = {
            "design_id": design_id,
            "sample_id": sample_id,
            "failure_seed": failure_seed,
            "error": repr(exc),
            "time": time.time(),
        }
        dump_json(failed_path, info)
        return {"ok": False, "path": str(out_path), "design_id": design_id, "sample_id": sample_id, "error": repr(exc)}


def read_run_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=True) as d:
        scalar = json.loads(str(np.asarray(d["scalar"]).item()))
        out = dict(scalar)
        out["path"] = str(path)
        out["force_valid_points"] = int(np.sum(d["force_valid_mask"]))
    return out


def aggregate_outputs(out_dir: Path, setup: dict, selected_design_ids: list[int], n_samples: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run_dir = out_dir / "runs"
    records = []
    for design_id in selected_design_ids:
        for sample_id in range(n_samples):
            p = run_dir / f"design_{design_id:03d}_sample_{sample_id:02d}.npz"
            if p.exists():
                records.append(read_run_npz(p))
            else:
                records.append({
                    "design_id": design_id,
                    "sample_id": sample_id,
                    "failure_seed": 3200000 + sample_id,
                    "run_status": "missing",
                    "energy": np.nan,
                    "peak_force": np.nan,
                    "final_peak": np.nan,
                    "n_failures": np.nan,
                    "max_cascade_size": np.nan,
                    "terminal_failed_member_fraction": np.nan,
                    "relative_volume_error": np.nan,
                    "runtime_s": np.nan,
                })

    run_df = pd.DataFrame(records)
    design_table = setup["design_table"].copy()
    design_table = design_table[design_table["design_id"].isin(selected_design_ids)].copy()

    design_rows = []
    lengths = setup["member_lengths"]
    member_layer = setup["member_layer"]
    family_code = setup["family_code"]
    family_names = setup["family_names"]
    area_m = setup["area_multipliers"]

    for design_id in selected_design_ids:
        sub = run_df[run_df["design_id"] == design_id].copy()
        valid = sub[~sub["run_status"].isin(["missing", "invalid", "timeout", "failed_output"])].copy()
        complete = sub[sub["run_status"] == "complete_physical_curve"].copy()
        singular = sub[sub["run_status"] == "singular_local_mechanism"].copy()
        near_zero = sub[sub["run_status"] == "near_zero"].copy()

        m = area_m[design_id]
        vol = m * lengths
        total_vol = float(np.sum(vol))

        layer_shares = {}
        for layer in sorted(set(member_layer.tolist())):
            layer_shares[f"volume_share_layer_{layer}"] = float(np.sum(vol[member_layer == layer]) / total_vol)

        family_shares = {}
        for code, fname in enumerate(family_names):
            family_shares[f"volume_share_family_{fname}"] = float(np.sum(vol[family_code == code]) / total_vol)

        meta_row = design_table[design_table["design_id"] == design_id]
        meta = meta_row.iloc[0].to_dict() if len(meta_row) else {}
        energies = complete["energy"].dropna().values

        design_rows.append({
            "design_id": int(design_id),
            "design_name": meta.get("design_name", f"design_{design_id:03d}"),
            "category": meta.get("category", "unknown"),
            "description": meta.get("description", ""),
            "configured_samples": int(n_samples),
            "completed_or_recorded": int(len(valid)),
            "missing_count": int((sub["run_status"] == "missing").sum()),
            "complete_physical_curve_count": int(len(complete)),
            "singular_local_mechanism_count": int(len(singular)),
            "singular_local_mechanism_rate": float(len(singular) / max(n_samples, 1)),
            "near_zero_count": int(len(near_zero)),
            "near_zero_rate": float(len(near_zero) / max(n_samples, 1)),
            "mean_energy": float(np.mean(energies)) if len(energies) else np.nan,
            "median_energy": float(np.median(energies)) if len(energies) else np.nan,
            "p10_energy": float(np.percentile(energies, 10)) if len(energies) else np.nan,
            "std_energy": float(np.std(energies)) if len(energies) else np.nan,
            "mean_peak_force": float(complete["peak_force"].mean()) if len(complete) else np.nan,
            "mean_final_peak_complete": float(complete["final_peak"].mean()) if len(complete) else np.nan,
            "mean_failures_per_run": float(complete["n_failures"].mean()) if len(complete) else np.nan,
            "mean_max_cascade": float(complete["max_cascade_size"].mean()) if len(complete) else np.nan,
            "mean_terminal_failed_member_fraction": float(complete["terminal_failed_member_fraction"].mean()) if len(complete) else np.nan,
            "area_min": float(np.min(m)),
            "area_median": float(np.median(m)),
            "area_max": float(np.max(m)),
            "rel_volume_error": float((np.sum(m * lengths) - np.sum(lengths)) / np.sum(lengths)),
            **layer_shares,
            **family_shares,
        })

    design_df = pd.DataFrame(design_rows)

    # Paired metrics vs uniform and vs design 002 when available.
    pair_rows = []
    uniform = run_df[(run_df["design_id"] == 0) & (run_df["run_status"] == "complete_physical_curve")][["sample_id", "energy"]].rename(columns={"energy": "uniform_energy"})
    vertical = run_df[(run_df["design_id"] == 2) & (run_df["run_status"] == "complete_physical_curve")][["sample_id", "energy"]].rename(columns={"energy": "vertical_energy"})

    for design_id in selected_design_ids:
        if design_id == 0:
            continue
        sub = run_df[(run_df["design_id"] == design_id) & (run_df["run_status"] == "complete_physical_curve")][["sample_id", "energy"]].copy()
        sub = sub.merge(uniform, on="sample_id", how="inner")
        sub = sub.merge(vertical, on="sample_id", how="left")
        if len(sub) == 0:
            pair_rows.append({"design_id": design_id, "paired_count_vs_uniform": 0})
            continue
        ratios = sub["energy"] / sub["uniform_energy"]
        deltas = sub["energy"] - sub["uniform_energy"]
        row = {
            "design_id": int(design_id),
            "paired_count_vs_uniform": int(len(sub)),
            "mean_ratio_vs_uniform": float(ratios.mean()),
            "median_ratio_vs_uniform": float(ratios.median()),
            "p10_ratio_vs_uniform": float(np.percentile(ratios, 10)),
            "worst_ratio_vs_uniform": float(ratios.min()),
            "mean_delta_vs_uniform": float(deltas.mean()),
            "median_delta_vs_uniform": float(deltas.median()),
            "beats_uniform_count": int((ratios > 1.0).sum()),
            "beats_uniform_fraction": float((ratios > 1.0).mean()),
            "std_ratio_vs_uniform": float(ratios.std(ddof=0)),
        }
        if sub["vertical_energy"].notna().any():
            sub2 = sub[sub["vertical_energy"].notna()].copy()
            ratios2 = sub2["energy"] / sub2["vertical_energy"]
            row.update({
                "paired_count_vs_verticals": int(len(sub2)),
                "median_ratio_vs_verticals": float(ratios2.median()),
                "p10_ratio_vs_verticals": float(np.percentile(ratios2, 10)),
                "beats_verticals_count": int((ratios2 > 1.0).sum()),
                "beats_verticals_fraction": float((ratios2 > 1.0).mean()),
            })
        row["robust_score"] = float(row["median_ratio_vs_uniform"] - 0.5 * row["std_ratio_vs_uniform"] - 0.25 * float(design_df.loc[design_df["design_id"] == design_id, "singular_local_mechanism_rate"].iloc[0]))
        pair_rows.append(row)
    pairwise_df = pd.DataFrame(pair_rows)

    # Attach names/categories.
    for df in [pairwise_df]:
        if len(df):
            df["design_name"] = df["design_id"].map(design_table.set_index("design_id")["design_name"].to_dict())
            df["category"] = df["design_id"].map(design_table.set_index("design_id")["category"].to_dict())
    if len(pairwise_df):
        pairwise_df = pairwise_df.sort_values(["median_ratio_vs_uniform", "p10_ratio_vs_uniform"], ascending=False, na_position="last")

    return run_df, design_df, pairwise_df


def make_figures(out_dir: Path, setup: dict, run_df: pd.DataFrame, design_df: pd.DataFrame, pairwise_df: pd.DataFrame, selected_design_ids: list[int]) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    if len(pairwise_df):
        top = pairwise_df.dropna(subset=["median_ratio_vs_uniform"]).head(12)
        plt.figure(figsize=(10, 5))
        labels = [f"{int(r.design_id):03d}" for _, r in top.iterrows()]
        plt.bar(labels, top["median_ratio_vs_uniform"])
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Design ID")
        plt.ylabel("Median energy ratio vs uniform")
        plt.title("Paired median energy ratio vs uniform")
        plt.tight_layout()
        plt.savefig(fig_dir / "paired_median_ratio_vs_uniform.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(labels, top["p10_ratio_vs_uniform"])
        plt.axhline(1.0, linestyle="--")
        plt.xlabel("Design ID")
        plt.ylabel("P10 energy ratio vs uniform")
        plt.title("Paired lower-tail energy ratio vs uniform")
        plt.tight_layout()
        plt.savefig(fig_dir / "paired_p10_ratio_vs_uniform.png", dpi=160)
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.bar(labels, top["beats_uniform_fraction"])
        plt.xlabel("Design ID")
        plt.ylabel("Fraction of paired seeds beating uniform")
        plt.title("Paired win fraction vs uniform")
        plt.tight_layout()
        plt.savefig(fig_dir / "paired_win_fraction_vs_uniform.png", dpi=160)
        plt.close()

    # Force overlays for selected important designs from completed run files.
    run_dir = out_dir / "runs"
    important = [d for d in [0, 2, 23, 26, 29, 46, 61] if d in selected_design_ids]
    for design_id in important:
        files = sorted(run_dir.glob(f"design_{design_id:03d}_sample_*.npz"))
        if not files:
            continue
        plt.figure(figsize=(8, 5))
        for p in files:
            with np.load(p, allow_pickle=True) as d:
                forces = d["forces"]
                disps = d["displacements"]
                mask = d["force_valid_mask"]
                n = min(len(forces), len(mask), len(disps))
                valid = mask[:n]
                if valid.any():
                    plt.plot(disps[:n][valid], forces[:n][valid], alpha=0.45)
        plt.xlabel("Displacement")
        plt.ylabel("Force")
        name = design_df.loc[design_df["design_id"] == design_id, "design_name"].iloc[0] if design_id in set(design_df["design_id"]) else f"design_{design_id:03d}"
        plt.title(f"Force-displacement overlays: {design_id:03d} {name}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"force_overlay_design_{design_id:03d}.png", dpi=160)
        plt.close()

    # Volume share by family / layer for top designs
    if len(pairwise_df):
        top_ids = [0, 2] + [int(x) for x in pairwise_df["design_id"].head(5).tolist()]
        top_ids = list(dict.fromkeys([x for x in top_ids if x in selected_design_ids]))
    else:
        top_ids = selected_design_ids[:5]

    layer_cols = [c for c in design_df.columns if c.startswith("volume_share_layer_")]
    family_cols = [c for c in design_df.columns if c.startswith("volume_share_family_")]
    for cols, name, title in [
        (layer_cols, "volume_share_by_layer_top_designs.png", "Volume share by layer"),
        (family_cols, "volume_share_by_family_top_designs.png", "Volume share by family"),
    ]:
        if not cols:
            continue
        sub = design_df[design_df["design_id"].isin(top_ids)].set_index("design_id")[cols]
        if len(sub):
            sub.plot(kind="bar", stacked=True, figsize=(10, 5))
            plt.ylabel("Volume share")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fig_dir / name, dpi=160)
            plt.close()

    # Area histograms for top designs
    area_m = setup["area_multipliers"]
    if len(top_ids):
        plt.figure(figsize=(9, 5))
        for design_id in top_ids[:6]:
            plt.hist(area_m[design_id], bins=30, alpha=0.35, label=f"{design_id:03d}")
        plt.xlabel("Area multiplier")
        plt.ylabel("Count")
        plt.title("Area multiplier histograms")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "area_multiplier_histograms_top_designs.png", dpi=160)
        plt.close()


def write_outputs(out_dir: Path, setup: dict, run_df: pd.DataFrame, design_df: pd.DataFrame, pairwise_df: pd.DataFrame, selected_design_ids: list[int], n_samples: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    run_csv = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_run_table.csv"
    design_csv = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_design_table.csv"
    pairwise_csv = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_pairwise_table.csv"
    summary_json = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_followup_summary.json"
    report_md = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_followup_report.md"
    designs_npz = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_designs.npz"
    targets_npz = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_targets.npz"

    run_df.to_csv(run_csv, index=False)
    design_df.to_csv(design_csv, index=False)
    pairwise_df.to_csv(pairwise_csv, index=False)

    selected_area = setup["area_multipliers"][selected_design_ids]
    np.savez_compressed(
        designs_npz,
        selected_design_ids=np.asarray(selected_design_ids, dtype=np.int32),
        area_multipliers=selected_area.astype(np.float32),
        member_lengths=setup["member_lengths"].astype(np.float64),
        member_layer=setup["member_layer"].astype(np.int16),
        family_code=setup["family_code"].astype(np.int16),
        family_names=np.asarray(setup["family_names"], dtype=object),
        nodes=setup["nodes"].astype(np.float64),
        members=setup["members"].astype(np.int64),
    )

    # Compact target arrays from completed files.
    completed = run_df[~run_df["run_status"].isin(["missing", "invalid", "timeout", "failed_output"])].copy()
    n = len(completed)
    max_steps = N_STEPS
    force_curves = np.full((n, max_steps), np.nan, dtype=np.float64)
    force_valid_mask = np.zeros((n, max_steps), dtype=bool)
    design_ids = np.zeros(n, dtype=np.int16)
    sample_ids = np.zeros(n, dtype=np.int16)
    energies = np.zeros(n, dtype=np.float64)
    status = []
    for i, (_, rec) in enumerate(completed.iterrows()):
        p = Path(rec["path"])
        with np.load(p, allow_pickle=True) as d:
            forces = np.asarray(d["forces"], dtype=np.float64)
            mask = np.asarray(d["force_valid_mask"], dtype=bool)
            m = min(len(forces), max_steps)
            force_curves[i, :m] = forces[:m]
            force_valid_mask[i, :min(len(mask), max_steps)] = mask[:max_steps]
        design_ids[i] = int(rec["design_id"])
        sample_ids[i] = int(rec["sample_id"])
        energies[i] = float(rec["energy"])
        status.append(str(rec["run_status"]))

    np.savez_compressed(
        targets_npz,
        design_ids=design_ids,
        sample_ids=sample_ids,
        force_curves=force_curves,
        force_valid_mask=force_valid_mask,
        energy=energies,
        run_status=np.asarray(status, dtype=object),
    )

    completed_count = int((run_df["run_status"] == "complete_physical_curve").sum())
    singular_count = int((run_df["run_status"] == "singular_local_mechanism").sum())
    near_zero_count = int((run_df["run_status"] == "near_zero").sum())
    missing_count = int((run_df["run_status"] == "missing").sum())
    failed_count = int((run_df["run_status"].isin(["invalid", "timeout", "failed_output"])).sum())

    top_pair = pairwise_df.head(8).to_dict(orient="records") if len(pairwise_df) else []
    summary = {
        "status": "experimental",
        "geometry": GEOMETRY_NAME,
        "shape": SHAPE,
        "scale": SCALE,
        "initial_prune_frac": INITIAL_PRUNE_FRAC,
        "selected_design_ids": selected_design_ids,
        "n_samples": n_samples,
        "target_runs": int(len(selected_design_ids) * n_samples),
        "complete_physical_curves": completed_count,
        "singular_local_mechanism_curves": singular_count,
        "near_zero_curves": near_zero_count,
        "missing_runs": missing_count,
        "failed_or_timeout_records": failed_count,
        "top_pairwise_designs": top_pair,
    }
    dump_json(summary_json, summary)

    if len(pairwise_df):
        top_lines = pairwise_df[[
            "design_id", "design_name", "category", "paired_count_vs_uniform",
            "median_ratio_vs_uniform", "p10_ratio_vs_uniform", "beats_uniform_count",
            "beats_uniform_fraction", "median_ratio_vs_verticals", "beats_verticals_fraction"
        ]].head(12).to_markdown(index=False)
    else:
        top_lines = "No pairwise rows available."

    report = f"""# Paired-Seed Area-Conditioned Follow-Up

Status: experimental, not official.

## Setting

- Geometry: `{GEOMETRY_NAME}`
- Weibull shape: `{SHAPE}`
- Weibull scale: `{SCALE}`
- Initial prune fraction: `{INITIAL_PRUNE_FRAC}`
- Max displacement: `{MAX_DISP}`
- Steps: `{N_STEPS}`

## Run accounting

- Selected designs: `{len(selected_design_ids)}`
- Samples/design: `{n_samples}`
- Target runs: `{len(selected_design_ids) * n_samples}`
- Complete physical curves: `{completed_count}`
- Singular/local-mechanism curves: `{singular_count}`
- Near-zero curves: `{near_zero_count}`
- Missing runs: `{missing_count}`
- Failed/timeout records: `{failed_count}`

## Top paired rankings

{top_lines}

## Interpretation notes

Use design `000` as the paired uniform reference. The main ranking is based on paired energy ratio under common failure-strain seeds.

This is still a screening run. Do not treat it as an official dataset or final robust optimization result.
"""
    report_md.write_text(report, encoding="utf-8")

    make_figures(out_dir, setup, run_df, design_df, pairwise_df, selected_design_ids)

    # Zip figures and final artifacts.
    fig_zip = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_figures.zip"
    with zipfile.ZipFile(fig_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted((out_dir / "figures").glob("*")):
            if p.is_file():
                z.write(p, arcname=p.name)

    artifact_zip = out_dir / "nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
    if artifact_zip.exists():
        artifact_zip.unlink()
    with zipfile.ZipFile(artifact_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file() and p != artifact_zip:
                z.write(p, arcname=str(p.relative_to(out_dir)))

    return artifact_zip


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--design-artifacts", required=True, type=Path)
    p.add_argument("--candidate-package", type=Path, default=None, help="Kept for provenance/reference; not required by this runner.")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--design-ids", default="000,001,002,023,026,029,046,047,061")
    p.add_argument("--n-samples", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--resume", action="store_true", default=True)
    args = p.parse_args()

    setup = load_pilot_designs(args.design_artifacts)
    selected_design_ids = parse_design_ids(args.design_ids)
    n_samples = int(args.n_samples)
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "runs").mkdir(exist_ok=True)
    (out_dir / "sidecars").mkdir(exist_ok=True)
    (out_dir / "failures").mkdir(exist_ok=True)

    # Save selected design metadata upfront.
    selected_meta = setup["design_table"][setup["design_table"]["design_id"].isin(selected_design_ids)].copy()
    selected_meta.to_csv(out_dir / "selected_designs.csv", index=False)

    # Prepare tasks; skip completed valid files when resume is on.
    tasks = []
    design_meta_by_id = setup["design_table"].set_index("design_id").to_dict(orient="index")
    for design_id in selected_design_ids:
        if design_id < 0 or design_id >= setup["area_multipliers"].shape[0]:
            raise ValueError(f"design_id {design_id} not present in area multipliers")
        for sample_id in range(n_samples):
            run_path = out_dir / "runs" / f"design_{design_id:03d}_sample_{sample_id:02d}.npz"
            if args.resume and run_path.exists():
                try:
                    read_run_npz(run_path)
                    continue
                except Exception:
                    run_path.unlink(missing_ok=True)
            failure_seed = 3200000 + sample_id
            tasks.append({
                "design_id": int(design_id),
                "sample_id": int(sample_id),
                "failure_seed": int(failure_seed),
                "area_multipliers": setup["area_multipliers"][design_id].astype(float),
                "member_lengths": setup["member_lengths"].astype(float),
                "design_meta": design_meta_by_id.get(design_id, {}),
                "out_path": str(run_path),
                "status_path": str(out_dir / "sidecars" / f"design_{design_id:03d}_sample_{sample_id:02d}.status.json"),
                "failed_path": str(out_dir / "failures" / f"design_{design_id:03d}_sample_{sample_id:02d}.failed.json"),
            })

    print(f"Selected designs: {selected_design_ids}")
    print(f"Samples/design: {n_samples}")
    print(f"Missing tasks to run: {len(tasks)}")
    print(f"Workers: {args.workers}")
    sys.stdout.flush()

    results = []
    if tasks:
        if args.workers <= 1:
            for t in tasks:
                res = run_one_task(t)
                results.append(res)
                print(res)
                sys.stdout.flush()
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                futs = [ex.submit(run_one_task, t) for t in tasks]
                for i, fut in enumerate(as_completed(futs), 1):
                    res = fut.result()
                    results.append(res)
                    print(f"[{i}/{len(futs)}] {res}")
                    sys.stdout.flush()

    dump_json(out_dir / "execution_results.json", results)

    run_df, design_df, pairwise_df = aggregate_outputs(out_dir, setup, selected_design_ids, n_samples)
    artifact_zip = write_outputs(out_dir, setup, run_df, design_df, pairwise_df, selected_design_ids, n_samples)

    print("\nDONE")
    print(f"Output directory: {out_dir}")
    print(f"Artifact ZIP: {artifact_zip}")
    if len(pairwise_df):
        print("\nTop paired designs:")
        cols = ["design_id", "design_name", "median_ratio_vs_uniform", "p10_ratio_vs_uniform", "beats_uniform_count", "beats_uniform_fraction"]
        print(pairwise_df[cols].head(10).to_string(index=False))
    print("\nRun accounting:")
    print(run_df["run_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
