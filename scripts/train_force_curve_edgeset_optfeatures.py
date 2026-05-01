#!/usr/bin/env python3
"""
Gradient-ready feature variants for the LATTICE EdgeSetMLP force-curve surrogate.

This script intentionally avoids elastic-demand solves, hard rank/percentile features,
and post-run target features. The goal is to test feature sets that are easy to
reimplement as differentiable PyTorch feature functions for later area optimization.

Feature sets:
  opt_basic:
    static geometry/topology + failure strain + direct area/volume features
  opt_area_context:
    opt_basic + area relative to layer/family/layer-family means/z-scores

It reuses the existing train_model() from scripts/train_force_curve_edgeset_fe.py,
so the architecture, masking, split policy, and loss are the same as the current
feature-engineered EdgeSetMLP trainer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Reuse the existing trainer infrastructure from the current FE script.
from train_force_curve_edgeset_fe import (
    set_seed,
    load_npz_from_zip,
    make_design_splits,
    train_model,
)

EPS = 1e-8


def group_mean_std(values: np.ndarray, group_ids: np.ndarray):
    """Return per-edge group mean/std for values shaped (B,E)."""
    values = np.asarray(values, dtype=np.float32)
    group_ids = np.asarray(group_ids)
    mean = np.zeros_like(values, dtype=np.float32)
    std = np.zeros_like(values, dtype=np.float32)
    for g in np.unique(group_ids):
        m = group_ids == g
        vals = values[:, m]
        mu = vals.mean(axis=1, keepdims=True)
        sig = vals.std(axis=1, keepdims=True)
        mean[:, m] = mu
        std[:, m] = np.maximum(sig, EPS)
    return mean, std


def build_features(designs, targets, failure_pack, feature_set="opt_basic"):
    if feature_set not in {"opt_basic", "opt_area_context"}:
        raise ValueError("feature_set must be opt_basic or opt_area_context")

    selected_design_ids = np.asarray(designs["selected_design_ids"]).astype(int)
    area = np.asarray(designs["area_multipliers"], dtype=np.float32)  # (D,E)
    lengths = np.asarray(designs["member_lengths"], dtype=np.float32)
    member_layer = np.asarray(designs["member_layer"]).astype(int)
    family_code = np.asarray(designs["family_code"]).astype(int)
    nodes = np.asarray(designs["nodes"], dtype=np.float32)
    members = np.asarray(designs["members"]).astype(int)

    run_design_ids = np.asarray(targets["design_ids"]).astype(int)
    sample_ids = np.asarray(targets["sample_ids"]).astype(int)
    force_curves = np.asarray(targets["force_curves"], dtype=np.float32)
    force_valid_mask = np.asarray(targets["force_valid_mask"]).astype(bool)
    run_status = np.asarray(targets["run_status"]).astype(str)

    fs_design_ids = np.asarray(failure_pack["design_ids"]).astype(int)
    fs_sample_ids = np.asarray(failure_pack["sample_ids"]).astype(int)
    fs = np.asarray(failure_pack["failure_strains"], dtype=np.float32)

    # Align failure strains to target rows by (design_id, sample_id).
    fs_index = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(fs_design_ids, fs_sample_ids))}
    fs_rows = np.array([fs_index[(int(d), int(s))] for d, s in zip(run_design_ids, sample_ids)], dtype=int)
    fs = fs[fs_rows]

    did_to_design_index = {int(d): i for i, d in enumerate(selected_design_ids)}
    design_rows_all = np.array([did_to_design_index[int(d)] for d in run_design_ids], dtype=int)

    # Keep same valid statuses as the FE script: complete physical curves and censored/local-mechanism curves with masks.
    valid_points = force_valid_mask.sum(axis=1)
    keep = np.isin(run_status, ["complete_physical_curve", "singular_local_mechanism"]) & (valid_points >= 2)
    keep_idx = np.flatnonzero(keep)

    n_runs = len(keep_idx)
    E = len(lengths)

    # Static geometry/topology features.
    a = members[:, 0]
    b = members[:, 1]
    vec = nodes[b] - nodes[a]
    L = np.linalg.norm(vec, axis=1).astype(np.float32)
    dirs = (vec / np.maximum(L[:, None], EPS)).astype(np.float32)
    abs_dirs = np.abs(dirs)
    mids = ((nodes[a] + nodes[b]) / 2.0).astype(np.float32)
    mids_center = mids - nodes.mean(axis=0, keepdims=True)
    mids_scale = np.maximum(nodes.std(axis=0, keepdims=True), EPS)
    mids_norm = (mids_center / mids_scale).astype(np.float32)
    z = nodes[:, 2]
    edge_z_span = np.abs(nodes[b, 2] - nodes[a, 2]).astype(np.float32)[:, None]
    top_touch = ((np.isclose(nodes[a, 2], z.max())) | (np.isclose(nodes[b, 2], z.max()))).astype(np.float32)[:, None]
    bottom_touch = ((np.isclose(nodes[a, 2], z.min())) | (np.isclose(nodes[b, 2], z.min()))).astype(np.float32)[:, None]
    radius = np.linalg.norm(mids_center[:, :2], axis=1, keepdims=True).astype(np.float32)

    length_feat = np.log(lengths + EPS).astype(np.float32)[:, None]
    fam_n = int(family_code.max()) + 1
    fam_oh = np.eye(fam_n, dtype=np.float32)[family_code]
    layer_n = int(member_layer.max()) + 1
    layer_oh = np.eye(layer_n, dtype=np.float32)[member_layer]

    static_edge = np.concatenate([
        length_feat,
        fam_oh,
        layer_oh,
        mids_norm,
        dirs,
        abs_dirs,
        edge_z_span,
        radius,
        top_touch,
        bottom_touch,
    ], axis=1).astype(np.float32)

    design_rows = design_rows_all[keep_idx]
    run_area = area[design_rows].astype(np.float32)
    run_fs = fs[keep_idx].astype(np.float32)
    volume = run_area * lengths[None, :]

    area_feats = np.stack([
        run_area,
        np.log(np.clip(run_area, EPS, None)),
        volume,
        np.log(np.clip(volume, EPS, None)),
    ], axis=-1).astype(np.float32)

    fs_log = np.log10(np.clip(run_fs, 1e-12, None))[:, :, None].astype(np.float32)
    geom_b = np.broadcast_to(static_edge[None, :, :], (n_runs, E, static_edge.shape[1])).copy()

    parts = [geom_b, area_feats, fs_log]
    feature_names = (
        ["log_length"] +
        [f"family_{i}" for i in range(fam_n)] +
        [f"layer_{i}" for i in range(layer_n)] +
        ["mid_x", "mid_y", "mid_z",
         "dir_x", "dir_y", "dir_z",
         "abs_dir_x", "abs_dir_y", "abs_dir_z",
         "z_span", "xy_radius", "top_touch", "bottom_touch",
         "area_multiplier", "log_area_multiplier",
         "volume_contribution", "log_volume_contribution",
         "log10_failure_strain"]
    )

    if feature_set == "opt_area_context":
        layer_mean, layer_std = group_mean_std(run_area, member_layer)
        fam_mean, fam_std = group_mean_std(run_area, family_code)
        layer_family_group = member_layer * max(1, fam_n) + family_code
        lf_mean, lf_std = group_mean_std(run_area, layer_family_group)

        vol_layer_mean, _ = group_mean_std(volume, member_layer)
        vol_fam_mean, _ = group_mean_std(volume, family_code)
        vol_lf_mean, _ = group_mean_std(volume, layer_family_group)

        area_layer_ratio = run_area / np.clip(layer_mean, EPS, None)
        area_fam_ratio = run_area / np.clip(fam_mean, EPS, None)
        area_lf_ratio = run_area / np.clip(lf_mean, EPS, None)

        context_feats = np.stack([
            area_layer_ratio,
            np.log(np.clip(area_layer_ratio, EPS, None)),
            (run_area - layer_mean) / np.clip(layer_std, EPS, None),

            area_fam_ratio,
            np.log(np.clip(area_fam_ratio, EPS, None)),
            (run_area - fam_mean) / np.clip(fam_std, EPS, None),

            area_lf_ratio,
            np.log(np.clip(area_lf_ratio, EPS, None)),
            (run_area - lf_mean) / np.clip(lf_std, EPS, None),

            volume / np.clip(vol_layer_mean, EPS, None),
            volume / np.clip(vol_fam_mean, EPS, None),
            volume / np.clip(vol_lf_mean, EPS, None),
        ], axis=-1).astype(np.float32)

        parts.append(context_feats)
        feature_names += [
            "area_rel_layer_mean", "log_area_rel_layer_mean", "area_z_layer",
            "area_rel_family_mean", "log_area_rel_family_mean", "area_z_family",
            "area_rel_layer_family_mean", "log_area_rel_layer_family_mean", "area_z_layer_family",
            "volume_rel_layer_mean", "volume_rel_family_mean", "volume_rel_layer_family_mean",
        ]

    edge_features = np.concatenate(parts, axis=-1).astype(np.float32)
    edge_features_all_finite = bool(np.isfinite(edge_features).all())

    y_curve = np.nan_to_num(force_curves[keep_idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_mask = force_valid_mask[keep_idx].astype(bool)

    return {
        "edge_features": edge_features,
        "y_curve": y_curve,
        "y_mask": y_mask,
        "design_ids": run_design_ids[keep_idx],
        "sample_ids": sample_ids[keep_idx],
        "run_status": run_status[keep_idx],
        "feature_info": {
            "feature_set": feature_set,
            "edge_feature_names": feature_names,
            "n_edge_features": int(edge_features.shape[-1]),
            "n_added_vs_opt_basic": int(edge_features.shape[-1] - 38) if feature_set == "opt_area_context" else 0,
            "n_kept_runs": int(n_runs),
            "n_edges": int(E),
            "n_steps": int(y_curve.shape[1]),
            "edge_features_all_finite": edge_features_all_finite,
            "status_counts_kept": {str(k): int(v) for k, v in zip(*np.unique(run_status[keep_idx], return_counts=True))},
            "leakage_note": "Only static geometry, area multipliers, volume contribution, and failure strains are used. No elastic solve, ranks, force, energy, cascade, or status inputs.",
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-zip", required=True, type=Path)
    ap.add_argument("--failure-strains", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--feature-set", choices=["opt_basic", "opt_area_context"], default="opt_basic")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--patience", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.0012)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--energy-loss-weight", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260428)
    ap.add_argument("--device", default="")
    ap.add_argument("--print-every", type=int, default=25)
    args = ap.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    designs, design_member = load_npz_from_zip(args.artifact_zip, "area_conditioned_paired_designs.npz")
    targets, targets_member = load_npz_from_zip(args.artifact_zip, "area_conditioned_paired_targets.npz")
    failure_pack = np.load(args.failure_strains, allow_pickle=True)

    print(f"Loaded designs from {design_member}")
    print(f"Loaded targets from {targets_member}")
    print(f"Loaded failure strains {args.failure_strains}")
    print(f"Feature set: {args.feature_set}")

    data = build_features(designs, targets, failure_pack, feature_set=args.feature_set)
    splits = make_design_splits(data["design_ids"], seed=args.seed, n_val=8, n_test=8)

    (args.out / "feature_info.json").write_text(json.dumps(data["feature_info"], indent=2), encoding="utf-8")
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = {
        "artifact_zip": str(args.artifact_zip),
        "failure_strains": str(args.failure_strains),
        "target": "force_curve_100",
        "feature_set": args.feature_set,
        "n_examples_kept": int(len(data["y_curve"])),
        "feature_info": data["feature_info"],
    }
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset summary:")
    print(json.dumps(summary, indent=2))

    if not data["feature_info"]["edge_features_all_finite"]:
        raise RuntimeError("Non-finite edge features detected.")

    train_model(data, splits, args.out, args)


if __name__ == "__main__":
    main()
