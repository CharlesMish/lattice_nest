#!/usr/bin/env python3
"""
Feature-engineered EdgeSetMLP force-curve surrogate for LATTICE.

Adds pre-run, non-leaky edge features:
  - geometry: midpoint xyz, unit direction, z-span, radius
  - design: area, log area, volume contribution
  - elastic demand from a tiny linear elastic solve per design
  - failure margin features: failure_strain vs elastic strain demand

Target:
  force_curves[100] with force_valid_mask

Split:
  by design_id, never by run

Loss:
  masked per-step force MSE with per-step normalization
  + optional energy-from-curve loss
"""
from __future__ import annotations

import argparse
import io
import json
import math
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_STEPS = 100
DISPLACEMENTS = np.asarray([0.0005 * (i + 1) for i in range(N_STEPS)], dtype=np.float32)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_member(names, suffix):
    hits = [n for n in names if n.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"Could not find member ending with {suffix}")
    return sorted(hits, key=len)[0]


def load_npz_from_zip(zip_path: Path, suffix: str):
    with zipfile.ZipFile(zip_path) as z:
        name = find_member(z.namelist(), suffix)
        data = np.load(io.BytesIO(z.read(name)), allow_pickle=True)
        return {k: data[k] for k in data.files}, name


def make_design_splits(design_ids, seed=20260428, n_val=8, n_test=8):
    rng = np.random.default_rng(seed)
    ids = np.array(sorted(set(int(x) for x in design_ids)), dtype=np.int64)
    perm = rng.permutation(ids)
    test = np.sort(perm[:n_test])
    val = np.sort(perm[n_test:n_test+n_val])
    train = np.sort(perm[n_test+n_val:])
    return {
        "seed": seed,
        "train_design_ids": train.tolist(),
        "val_design_ids": val.tolist(),
        "test_design_ids": test.tolist(),
    }


def truss_elastic_strain_features(nodes, members, area, step_disp=0.0005):
    """Approximate initial elastic axial strain/force/energy under top z displacement.

    This is a pre-run feature, not leakage: it uses only geometry and area design.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    members = np.asarray(members, dtype=np.int64)
    area = np.asarray(area, dtype=np.float64)
    n_nodes = nodes.shape[0]
    ndim = 3
    n_dof = n_nodes * ndim

    a = members[:, 0]
    b = members[:, 1]
    vec = nodes[b] - nodes[a]
    L = np.linalg.norm(vec, axis=1)
    dirs = vec / np.maximum(L[:, None], 1e-12)

    K = np.zeros((n_dof, n_dof), dtype=np.float64)
    for e, (i, j) in enumerate(members):
        n = dirs[e]
        k3 = (area[e] / max(L[e], 1e-12)) * np.outer(n, n)
        di = np.arange(i * ndim, i * ndim + ndim)
        dj = np.arange(j * ndim, j * ndim + ndim)
        K[np.ix_(di, di)] += k3
        K[np.ix_(di, dj)] -= k3
        K[np.ix_(dj, di)] -= k3
        K[np.ix_(dj, dj)] += k3

    z = nodes[:, 2]
    bottom = np.flatnonzero(np.isclose(z, z.min()))
    top = np.flatnonzero(np.isclose(z, z.max()))

    known = []
    u_known_vals = []
    for node in bottom:
        for d in range(ndim):
            known.append(node * ndim + d)
            u_known_vals.append(0.0)
    for node in top:
        known.append(node * ndim + 2)
        u_known_vals.append(float(step_disp))

    known = np.asarray(known, dtype=np.int64)
    u_known_vals = np.asarray(u_known_vals, dtype=np.float64)
    all_dofs = np.arange(n_dof, dtype=np.int64)
    known_mask = np.zeros(n_dof, dtype=bool)
    known_mask[known] = True
    free = all_dofs[~known_mask]

    u = np.zeros(n_dof, dtype=np.float64)
    u[known] = u_known_vals

    Kff = K[np.ix_(free, free)]
    Kfk = K[np.ix_(free, known)]
    rhs = -Kfk @ u_known_vals

    # Mild regularization for numerical mechanisms; feature only.
    reg = 1e-10 * max(float(np.trace(Kff)) / max(Kff.shape[0], 1), 1.0)
    try:
        u_free = np.linalg.solve(Kff + reg * np.eye(Kff.shape[0]), rhs)
    except np.linalg.LinAlgError:
        u_free = np.linalg.lstsq(Kff + reg * np.eye(Kff.shape[0]), rhs, rcond=None)[0]
    u[free] = u_free
    U = u.reshape(n_nodes, ndim)

    du = U[b] - U[a]
    axial_strain = np.sum(du * dirs, axis=1) / np.maximum(L, 1e-12)
    axial_force_unitE = area * axial_strain
    elastic_energy_unitE = 0.5 * area * L * axial_strain**2

    return axial_strain.astype(np.float32), axial_force_unitE.astype(np.float32), elastic_energy_unitE.astype(np.float32)


class EdgeCurveDataset(Dataset):
    def __init__(self, edge_features, y_curve, y_mask, design_ids, sample_ids, indices):
        self.edge_features = torch.as_tensor(edge_features[indices], dtype=torch.float32)
        self.y_curve = torch.as_tensor(y_curve[indices], dtype=torch.float32)
        self.y_mask = torch.as_tensor(y_mask[indices], dtype=torch.float32)
        self.design_ids = np.asarray(design_ids)[indices]
        self.sample_ids = np.asarray(sample_ids)[indices]
        self.indices = np.asarray(indices)

    def __len__(self):
        return self.edge_features.shape[0]

    def __getitem__(self, idx):
        return (
            self.edge_features[idx],
            self.y_curve[idx],
            self.y_mask[idx],
            int(self.design_ids[idx]),
            int(self.sample_ids[idx]),
            int(self.indices[idx]),
        )


class EdgeSetCurveMLP(nn.Module):
    def __init__(self, edge_in_dim, seq_len=100, hidden=192, edge_hidden=192, dropout=0.0, pooling="mean_sum"):
        super().__init__()
        allowed = {
            "mean",
            "mean_sum",
            "mean_std",
            "mean_std_max",
            "mean_std_max_min",
            "mean_sum_std",
            "mean_sum_std_max",
            "mean_sum_std_max_min",
        }
        if pooling not in allowed:
            raise ValueError(f"Unknown pooling={pooling}. Allowed: {sorted(allowed)}")
        self.pooling = pooling

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
        )

        n_parts = {
            "mean": 1,
            "mean_sum": 2,
            "mean_std": 2,
            "mean_std_max": 3,
            "mean_std_max_min": 4,
            "mean_sum_std": 3,
            "mean_sum_std_max": 4,
            "mean_sum_std_max_min": 5,
        }[pooling]

        self.head = nn.Sequential(
            nn.Linear(n_parts * edge_hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )

    def _pool(self, h):
        E = h.shape[1]
        parts = [h.mean(dim=1)]

        if "sum" in self.pooling:
            parts.append(h.sum(dim=1) / math.sqrt(E))

        if "std" in self.pooling:
            parts.append(h.std(dim=1, unbiased=False))

        if "max" in self.pooling:
            parts.append(h.max(dim=1).values)

        if "min" in self.pooling:
            parts.append(h.min(dim=1).values)

        return torch.cat(parts, dim=-1)

    def forward(self, edge_x):
        h = self.edge_mlp(edge_x)
        g = self._pool(h)
        return self.head(g)

def masked_mse(pred, true, mask):
    return (((pred - true) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)


def energy_torch(curve, mask, disp, use_abs=True):
    f = torch.abs(curve) if use_abs else curve
    m = mask > 0.5
    pair = (m[:, 1:] & m[:, :-1]).float()
    dx = disp[1:] - disp[:-1]
    e = 0.5 * (f[:, 1:] + f[:, :-1]) * dx[None, :] * pair
    return e.sum(dim=1)


def energy_np(curve, mask, use_abs=True):
    out = np.zeros(curve.shape[0], dtype=np.float32)
    fcurve = np.abs(curve) if use_abs else curve
    for i in range(curve.shape[0]):
        valid = mask[i].astype(bool)
        if valid.sum() >= 2:
            out[i] = float(np.trapezoid(fcurve[i, valid], DISPLACEMENTS[valid]))
        elif valid.sum() == 1:
            out[i] = 0.0
        else:
            out[i] = np.nan
    return out


def masked_flat_metrics(true, pred, mask):
    valid = mask.astype(bool)
    t = true[valid]
    p = pred[valid]
    rmse = float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else float("nan")
    mae = float(np.mean(np.abs(p - t))) if len(t) else float("nan")
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) if len(t) else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot and ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "flat_r2": r2}


def per_step_r2(true, pred, mask):
    vals = []
    for j in range(true.shape[1]):
        valid = mask[:, j].astype(bool)
        if valid.sum() < 2:
            continue
        t = true[valid, j]
        p = pred[valid, j]
        ss_res = float(np.sum((p - t) ** 2))
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        if ss_tot > 0:
            vals.append(1.0 - ss_res / ss_tot)
    return float(np.mean(vals)) if vals else float("nan")


def scalar_metrics(true, pred):
    good = np.isfinite(true) & np.isfinite(pred)
    t = true[good]
    p = pred[good]
    rmse = float(np.sqrt(np.mean((p - t) ** 2))) if len(t) else float("nan")
    mae = float(np.mean(np.abs(p - t))) if len(t) else float("nan")
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - t.mean()) ** 2)) if len(t) else float("nan")
    r2 = 1.0 - ss_res / ss_tot if ss_tot and ss_tot > 0 else float("nan")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def build_features(designs, targets, failure_pack, include_elastic=True):
    selected_design_ids = np.asarray(designs["selected_design_ids"]).astype(int)
    area = np.asarray(designs["area_multipliers"], dtype=np.float32)
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

    fs_index = {(int(d), int(s)): i for i, (d, s) in enumerate(zip(fs_design_ids, fs_sample_ids))}
    fs_rows = np.array([fs_index[(int(d), int(s))] for d, s in zip(run_design_ids, sample_ids)], dtype=int)
    fs = fs[fs_rows]

    did_to_design_index = {int(d): i for i, d in enumerate(selected_design_ids)}
    design_rows_all = np.array([did_to_design_index[int(d)] for d in run_design_ids], dtype=int)

    valid_points = force_valid_mask.sum(axis=1)
    keep = np.isin(run_status, ["complete_physical_curve", "singular_local_mechanism"]) & (valid_points >= 2)
    keep_idx = np.flatnonzero(keep)

    n_runs = len(keep_idx)
    E = len(lengths)

    a = members[:, 0]
    b = members[:, 1]
    vec = nodes[b] - nodes[a]
    L = np.linalg.norm(vec, axis=1).astype(np.float32)
    dirs = (vec / np.maximum(L[:, None], 1e-12)).astype(np.float32)
    abs_dirs = np.abs(dirs)
    mids = ((nodes[a] + nodes[b]) / 2.0).astype(np.float32)
    mids_center = mids - nodes.mean(axis=0, keepdims=True)
    mids_scale = np.maximum(nodes.std(axis=0, keepdims=True), 1e-6)
    mids_norm = (mids_center / mids_scale).astype(np.float32)
    z = nodes[:, 2]
    edge_z_span = np.abs(nodes[b, 2] - nodes[a, 2]).astype(np.float32)[:, None]
    top_touch = ((np.isclose(nodes[a, 2], z.max())) | (np.isclose(nodes[b, 2], z.max()))).astype(np.float32)[:, None]
    bottom_touch = ((np.isclose(nodes[a, 2], z.min())) | (np.isclose(nodes[b, 2], z.min()))).astype(np.float32)[:, None]
    radius = np.linalg.norm(mids_center[:, :2], axis=1, keepdims=True).astype(np.float32)

    length_feat = np.log(lengths + 1e-8).astype(np.float32)[:, None]
    fam_n = int(family_code.max()) + 1
    fam_oh = np.eye(fam_n, dtype=np.float32)[family_code]
    layer_n = int(member_layer.max()) + 1
    layer_oh = np.eye(layer_n, dtype=np.float32)[member_layer]

    geom_edge = np.concatenate([
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

    # Per-design elastic features.
    elastic_by_design = []
    if include_elastic:
        print("Computing elastic-demand features per design...", flush=True)
        for di, did in enumerate(selected_design_ids):
            if di % 8 == 0:
                print(f"  elastic features {di+1}/{len(selected_design_ids)}", flush=True)
            strain, force, elast_e = truss_elastic_strain_features(nodes, members, area[di])
            abs_strain = np.abs(strain)
            elastic_by_design.append(np.stack([
                strain,
                abs_strain,
                np.log10(abs_strain + 1e-12),
                force,
                np.abs(force),
                np.log10(np.abs(force) + 1e-12),
                elast_e,
                np.log10(elast_e + 1e-20),
            ], axis=1).astype(np.float32))
        elastic_by_design = np.stack(elastic_by_design, axis=0)
    else:
        elastic_by_design = np.zeros((len(selected_design_ids), E, 0), dtype=np.float32)

    design_rows = design_rows_all[keep_idx]
    run_area = area[design_rows].astype(np.float32)
    run_fs = fs[keep_idx].astype(np.float32)
    run_elastic = elastic_by_design[design_rows]

    area_feats = np.stack([
        run_area,
        np.log(np.clip(run_area, 1e-8, None)),
        run_area * lengths[None, :],
        np.log(np.clip(run_area * lengths[None, :], 1e-8, None)),
    ], axis=-1).astype(np.float32)

    fs_log = np.log10(np.clip(run_fs, 1e-12, None))[:, :, None]
    if include_elastic:
        abs_strain = np.abs(run_elastic[:, :, 0])
        demand_ratio = abs_strain / np.clip(run_fs, 1e-12, None)
        margin = run_fs / np.clip(abs_strain, 1e-12, None)
        margin_feats = np.stack([
            demand_ratio,
            np.log10(demand_ratio + 1e-12),
            margin,
            np.log10(margin + 1e-12),
        ], axis=-1).astype(np.float32)
    else:
        margin_feats = np.zeros((n_runs, E, 0), dtype=np.float32)

    geom_b = np.broadcast_to(geom_edge[None, :, :], (n_runs, E, geom_edge.shape[1])).copy()
    edge_features = np.concatenate([geom_b, area_feats, fs_log, run_elastic, margin_feats], axis=-1).astype(np.float32)

    y_curve = np.nan_to_num(force_curves[keep_idx].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y_mask = force_valid_mask[keep_idx].astype(bool)

    feature_names = (
        ["log_length"] +
        [f"family_{i}" for i in range(fam_n)] +
        [f"layer_{i}" for i in range(layer_n)] +
        ["mid_x", "mid_y", "mid_z", "dir_x", "dir_y", "dir_z", "abs_dir_x", "abs_dir_y", "abs_dir_z",
         "z_span", "xy_radius", "top_touch", "bottom_touch",
         "area_multiplier", "log_area_multiplier", "volume_contribution", "log_volume_contribution",
         "log10_failure_strain"] +
        (["elastic_strain", "elastic_abs_strain", "elastic_log_abs_strain",
          "elastic_force", "elastic_abs_force", "elastic_log_abs_force",
          "elastic_energy", "elastic_log_energy",
          "demand_ratio", "log_demand_ratio", "failure_margin", "log_failure_margin"] if include_elastic else [])
    )

    return {
        "edge_features": edge_features,
        "y_curve": y_curve,
        "y_mask": y_mask,
        "design_ids": run_design_ids[keep_idx],
        "sample_ids": sample_ids[keep_idx],
        "run_status": run_status[keep_idx],
        "feature_info": {
            "edge_feature_names": feature_names,
            "n_edge_features": int(edge_features.shape[-1]),
            "n_kept_runs": int(n_runs),
            "n_edges": int(E),
            "n_steps": int(y_curve.shape[1]),
            "status_counts_kept": {str(k): int(v) for k, v in zip(*np.unique(run_status[keep_idx], return_counts=True))},
            "include_elastic": bool(include_elastic),
        },
    }


def compute_norms(edge_features, y_curve, y_mask, train_mask):
    edge_train = edge_features[train_mask]
    x_mu = edge_train.reshape(-1, edge_features.shape[-1]).mean(axis=0).astype(np.float32)
    x_sig = edge_train.reshape(-1, edge_features.shape[-1]).std(axis=0).astype(np.float32)
    x_sig[x_sig < 1e-8] = 1.0

    y_mu = np.zeros((1, y_curve.shape[1]), dtype=np.float32)
    y_sig = np.ones((1, y_curve.shape[1]), dtype=np.float32)
    for j in range(y_curve.shape[1]):
        valid = train_mask & y_mask[:, j]
        vals = y_curve[valid, j]
        if len(vals) > 0:
            y_mu[0, j] = vals.mean()
            s = vals.std()
            y_sig[0, j] = s if s > 1e-8 else 1.0

    true_e = energy_np(y_curve[train_mask], y_mask[train_mask], use_abs=True)
    e_mu = float(np.nanmean(true_e))
    e_sig = float(np.nanstd(true_e))
    if e_sig < 1e-8:
        e_sig = 1.0
    return x_mu, x_sig, y_mu, y_sig, e_mu, e_sig


def eval_model(model, loader, device, y_mu, y_sig):
    model.eval()
    y_mu_t = torch.as_tensor(y_mu, dtype=torch.float32, device=device)
    y_sig_t = torch.as_tensor(y_sig, dtype=torch.float32, device=device)
    preds, trues, masks, dids, sids = [], [], [], [], []
    with torch.no_grad():
        for x, y, mask, did, sid, _ in loader:
            x = x.to(device)
            pred_n = model(x)
            pred = (pred_n * y_sig_t + y_mu_t).cpu().numpy()
            true = (y.to(device) * y_sig_t + y_mu_t).cpu().numpy()
            preds.append(pred); trues.append(true); masks.append(mask.numpy().astype(bool))
            dids.append(np.asarray(did)); sids.append(np.asarray(sid))
    pred = np.concatenate(preds, axis=0)
    true = np.concatenate(trues, axis=0)
    mask = np.concatenate(masks, axis=0)
    did = np.concatenate(dids)
    sid = np.concatenate(sids)

    curve = masked_flat_metrics(true, pred, mask)
    curve["mean_per_step_r2"] = per_step_r2(true, pred, mask)
    e_true = energy_np(true, mask, use_abs=True)
    e_pred = energy_np(pred, mask, use_abs=True)
    em = scalar_metrics(e_true, e_pred)
    return {"curve": curve, "energy_from_curve": em, "pred_curve": pred, "true_curve": true, "mask": mask,
            "true_energy": e_true, "pred_energy": e_pred, "design_id": did, "sample_id": sid}


def train_model(data, splits, out_dir, args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print("device:", device, flush=True)

    design_ids = data["design_ids"]
    train_designs = set(splits["train_design_ids"])
    val_designs = set(splits["val_design_ids"])
    test_designs = set(splits["test_design_ids"])
    train_mask = np.array([int(d) in train_designs for d in design_ids])
    val_mask = np.array([int(d) in val_designs for d in design_ids])
    test_mask = np.array([int(d) in test_designs for d in design_ids])

    edge = data["edge_features"].copy()
    y_curve = data["y_curve"].copy()
    y_mask = data["y_mask"].copy()

    x_mu, x_sig, y_mu, y_sig, e_mu, e_sig = compute_norms(edge, y_curve, y_mask, train_mask)
    edge = (edge - x_mu[None, None, :]) / x_sig[None, None, :]
    y_norm = (y_curve - y_mu) / y_sig
    y_norm[~y_mask] = 0.0

    train_idx = np.flatnonzero(train_mask)
    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)

    train_ds = EdgeCurveDataset(edge, y_norm, y_mask, design_ids, data["sample_ids"], train_idx)
    val_ds = EdgeCurveDataset(edge, y_norm, y_mask, design_ids, data["sample_ids"], val_idx)
    test_ds = EdgeCurveDataset(edge, y_norm, y_mask, design_ids, data["sample_ids"], test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = EdgeSetCurveMLP(edge_in_dim=edge.shape[-1], seq_len=y_curve.shape[1],
                            hidden=args.hidden, edge_hidden=args.hidden, dropout=args.dropout,
                            pooling=args.pooling).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * 0.05)
    disp_t = torch.as_tensor(DISPLACEMENTS, dtype=torch.float32, device=device)
    y_mu_t = torch.as_tensor(y_mu, dtype=torch.float32, device=device)
    y_sig_t = torch.as_tensor(y_sig, dtype=torch.float32, device=device)

    best_val = float("inf")
    best_state = None
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, mask, *_ in train_loader:
            x = x.to(device); y = y.to(device); mask = mask.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(x)
            loss_force = masked_mse(pred, y, mask)
            loss = loss_force
            if args.energy_loss_weight > 0:
                pred_den = pred * y_sig_t + y_mu_t
                true_den = y * y_sig_t + y_mu_t
                e_pred = energy_torch(pred_den, mask, disp_t, use_abs=True)
                e_true = energy_torch(true_den, mask, disp_t, use_abs=True)
                loss_energy = torch.mean(((e_pred - e_true) / e_sig) ** 2)
                loss = loss + args.energy_loss_weight * loss_energy
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        scheduler.step()

        val_metrics = eval_model(model, val_loader, device, y_mu, y_sig)
        val_score = val_metrics["curve"]["rmse"]
        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "val_force_rmse": val_metrics["curve"]["rmse"],
            "val_force_flat_r2": val_metrics["curve"]["flat_r2"],
            "val_mean_per_step_r2": val_metrics["curve"]["mean_per_step_r2"],
            "val_energy_rmse": val_metrics["energy_from_curve"]["rmse"],
            "val_energy_r2": val_metrics["energy_from_curve"]["r2"],
            "lr": float(opt.param_groups[0]["lr"]),
        })
        if val_score < best_val:
            best_val = val_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if epoch == 1 or epoch % args.print_every == 0:
            print(f"[edge_curve_fe:{args.pooling}] epoch={epoch:04d} loss={np.mean(losses):.4e} "
                  f"val_force_rmse={val_metrics['curve']['rmse']:.4f} "
                  f"val_flat_r2={val_metrics['curve']['flat_r2']:.4f} "
                  f"val_step_r2={val_metrics['curve']['mean_per_step_r2']:.4f} "
                  f"val_energy_rmse={val_metrics['energy_from_curve']['rmse']:.4f} "
                  f"val_energy_r2={val_metrics['energy_from_curve']['r2']:.4f}", flush=True)
        if patience_left <= 0:
            print(f"[edge_curve_fe:{args.pooling}] early stop at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_state)

    train_metrics = eval_model(model, train_loader, device, y_mu, y_sig)
    val_metrics = eval_model(model, val_loader, device, y_mu, y_sig)
    test_metrics = eval_model(model, test_loader, device, y_mu, y_sig)

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    pred_rows = []
    for split_name, metrics in [("train", train_metrics), ("val", val_metrics), ("test", test_metrics)]:
        for te, pe, did, sid in zip(metrics["true_energy"], metrics["pred_energy"], metrics["design_id"], metrics["sample_id"]):
            pred_rows.append({"split": split_name, "design_id": int(did), "sample_id": int(sid),
                              "true_energy_from_curve": float(te), "pred_energy_from_curve": float(pe),
                              "energy_error": float(pe - te)})
    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / "energy_from_curve_predictions.csv", index=False)

    np.savez_compressed(
        out_dir / "test_curve_predictions.npz",
        design_ids=test_metrics["design_id"].astype(np.int16),
        sample_ids=test_metrics["sample_id"].astype(np.int16),
        true_curve=test_metrics["true_curve"].astype(np.float32),
        pred_curve=test_metrics["pred_curve"].astype(np.float32),
        force_valid_mask=test_metrics["mask"].astype(bool),
        true_energy=test_metrics["true_energy"].astype(np.float32),
        pred_energy=test_metrics["pred_energy"].astype(np.float32),
        displacements=DISPLACEMENTS,
    )

    plt.figure(figsize=(6, 6))
    plt.scatter(test_metrics["true_energy"], test_metrics["pred_energy"], alpha=0.7)
    lo = min(float(np.nanmin(test_metrics["true_energy"])), float(np.nanmin(test_metrics["pred_energy"])))
    hi = max(float(np.nanmax(test_metrics["true_energy"])), float(np.nanmax(test_metrics["pred_energy"])))
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True energy from curve")
    plt.ylabel("Predicted energy from curve")
    plt.title("Feature-engineered EdgeSet curve model: test energy parity")
    plt.tight_layout()
    plt.savefig(out_dir / "test_energy_from_curve_parity.png", dpi=160)
    plt.close()

    test_err = np.sqrt(np.mean(((test_metrics["pred_curve"] - test_metrics["true_curve"]) ** 2) * test_metrics["mask"], axis=1))
    order = np.argsort(test_err)
    choices = {"best": order[0], "median": order[len(order)//2], "worst": order[-1]}
    for label, i in choices.items():
        valid = test_metrics["mask"][i].astype(bool)
        plt.figure(figsize=(8, 5))
        plt.plot(DISPLACEMENTS[valid], test_metrics["true_curve"][i, valid], label="true")
        plt.plot(DISPLACEMENTS[valid], test_metrics["pred_curve"][i, valid], label="predicted")
        plt.xlabel("Displacement"); plt.ylabel("Force")
        plt.title(f"{label} test curve: design {int(test_metrics['design_id'][i]):03d}, sample {int(test_metrics['sample_id'][i]):02d}")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_dir / f"test_curve_{label}.png", dpi=160)
        plt.close()

    design_agg = pred_df[pred_df["split"] == "test"].groupby("design_id").agg(
        true_mean_energy=("true_energy_from_curve", "mean"),
        pred_mean_energy=("pred_energy_from_curve", "mean"),
        true_median_energy=("true_energy_from_curve", "median"),
        pred_median_energy=("pred_energy_from_curve", "median"),
        n=("true_energy_from_curve", "size"),
    ).reset_index()
    design_agg["mean_energy_error"] = design_agg["pred_mean_energy"] - design_agg["true_mean_energy"]
    design_agg.to_csv(out_dir / "test_design_aggregate_predictions.csv", index=False)

    metrics = {
        "model": "FeatureEngineeredEdgeSetCurveMLP",
        "pooling": args.pooling,
        "device": str(device),
        "energy_loss_weight": args.energy_loss_weight,
        "n_train": int(len(train_ds)),
        "n_val": int(len(val_ds)),
        "n_test": int(len(test_ds)),
        "train_design_ids": splits["train_design_ids"],
        "val_design_ids": splits["val_design_ids"],
        "test_design_ids": splits["test_design_ids"],
        "train": {"force": train_metrics["curve"], "energy_from_curve": train_metrics["energy_from_curve"]},
        "val": {"force": val_metrics["curve"], "energy_from_curve": val_metrics["energy_from_curve"]},
        "test": {"force": test_metrics["curve"], "energy_from_curve": test_metrics["energy_from_curve"]},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save({"state_dict": model.state_dict(), "metrics": metrics, "feature_info": data["feature_info"]}, out_dir / "model.pt")
    rows = []
    for split in ["train", "val", "test"]:
        rows.append({"split": split, **{f"force_{k}": v for k, v in metrics[split]["force"].items()},
                     **{f"energy_{k}": v for k, v in metrics[split]["energy_from_curve"].items()}})
    pd.DataFrame(rows).to_csv(out_dir / "metrics_summary.csv", index=False)
    print("\nFINAL METRICS")
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact-zip", required=True, type=Path)
    ap.add_argument("--failure-strains", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--patience", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--energy-loss-weight", type=float, default=0.2)
    ap.add_argument("--pooling", type=str, default="mean_sum", choices=["mean", "mean_sum", "mean_std", "mean_std_max", "mean_std_max_min", "mean_sum_std", "mean_sum_std_max", "mean_sum_std_max_min"])
    ap.add_argument("--no-elastic", action="store_true")
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

    data = build_features(designs, targets, failure_pack, include_elastic=not args.no_elastic)
    splits = make_design_splits(data["design_ids"], seed=args.seed, n_val=8, n_test=8)

    (args.out / "feature_info.json").write_text(json.dumps(data["feature_info"], indent=2), encoding="utf-8")
    (args.out / "splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")
    summary = {
        "artifact_zip": str(args.artifact_zip),
        "failure_strains": str(args.failure_strains),
        "target": "force_curve_100",
        "n_examples_kept": int(len(data["y_curve"])),
        "status_counts_kept": data["feature_info"]["status_counts_kept"],
        "feature_info": data["feature_info"],
    }
    (args.out / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset summary:")
    print(json.dumps(summary, indent=2))

    train_model(data, splits, args.out, args)


if __name__ == "__main__":
    main()
