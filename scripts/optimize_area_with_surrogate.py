#!/usr/bin/env python3
"""
Surrogate-gradient area optimization prototype for LATTICE.

Uses the trained gradient-ready opt_area_context EdgeSetMLP surrogate to optimize
per-member area multipliers m_i.

Intended model:
  outputs/edgeset_force_curve_opt_area_context_fullcombo_h512/model.pt

Important:
  This is a surrogate proposal tool, not proof of physical improvement.
  Exported designs should be simulated afterward.

Feature set:
  opt_area_context only:
    static geometry/topology + failure strain + area/log-area/volume
    + differentiable area context features

No elastic solve.
No rank/percentile features.
No target/post-run features.

Optimization:
  theta -> m_i in [0.25, 3.0] by sigmoid
  length-weighted volume penalty encourages sum(m_i L_i)=sum(L_i)
  objective maximizes mean predicted energy, optionally risk-penalized by std
"""
from __future__ import annotations

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

N_STEPS = 100
DISPLACEMENTS = torch.tensor([0.0005 * (i + 1) for i in range(N_STEPS)], dtype=torch.float32)
LO = 0.25
HI = 3.0
EPS = 1e-8


class EdgeSetCurveMLP(nn.Module):
    def __init__(self, edge_in_dim, seq_len=100, hidden=512, edge_hidden=512, dropout=0.0):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, edge_hidden),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(2 * edge_hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, edge_x):
        h = self.edge_mlp(edge_x)
        g = torch.cat([h.mean(dim=1), h.sum(dim=1) / math.sqrt(edge_x.shape[1])], dim=-1)
        return self.head(g)


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


def project_volume_clip_np(raw, lengths, lo=LO, hi=HI, max_iter=40):
    """Non-gradient final projection to exact bounded length-weighted volume."""
    raw = np.asarray(raw, dtype=np.float64).copy()
    L = np.asarray(lengths, dtype=np.float64)
    target = float(np.sum(L))
    raw = np.nan_to_num(raw, nan=1.0, posinf=hi, neginf=lo)
    raw = np.maximum(raw, 1e-12)
    fixed = np.zeros_like(raw, dtype=bool)
    m = raw.copy()
    for _ in range(max_iter):
        free = ~fixed
        fixed_vol = float(np.sum(m[fixed] * L[fixed]))
        free_vol_raw = float(np.sum(raw[free] * L[free]))
        if free_vol_raw <= 1e-15:
            m[free] = (target - fixed_vol) / max(float(np.sum(L[free])), 1e-15)
        else:
            scale = (target - fixed_vol) / free_vol_raw
            m[free] = raw[free] * scale
        low = free & (m < lo)
        high = free & (m > hi)
        if not (low.any() or high.any()):
            break
        m[low] = lo
        m[high] = hi
        fixed[low | high] = True
    m = np.clip(m, lo, hi)
    for _ in range(12):
        err = target - float(np.sum(m * L))
        if abs(err) / target < 1e-10:
            break
        free = m < hi - 1e-9 if err > 0 else m > lo + 1e-9
        denom = float(np.sum(L[free]))
        if denom <= 1e-15:
            break
        m[free] += err / denom
        m = np.clip(m, lo, hi)
    return m.astype(np.float32)


def inverse_sigmoid_area(m, lo=LO, hi=HI):
    x = (np.asarray(m, dtype=np.float32) - lo) / (hi - lo)
    x = np.clip(x, 1e-5, 1 - 1e-5)
    return np.log(x / (1 - x)).astype(np.float32)


def unique_failure_fields(failure_pack, n, seed, device):
    """Deduplicate by sample_id first, then take n fields."""
    fs = np.asarray(failure_pack["failure_strains"], dtype=np.float32)
    sample_ids = np.asarray(failure_pack["sample_ids"]).astype(int)
    seen = {}
    for i, sid in enumerate(sample_ids):
        if int(sid) not in seen:
            seen[int(sid)] = i
    idxs = np.array([seen[k] for k in sorted(seen.keys())], dtype=int)
    rng = np.random.default_rng(seed)
    if n < len(idxs):
        idxs = rng.choice(idxs, size=n, replace=False)
    else:
        idxs = idxs
    return torch.as_tensor(fs[idxs], dtype=torch.float32, device=device), idxs


def build_static_tensors(designs, device):
    nodes = torch.as_tensor(np.asarray(designs["nodes"], dtype=np.float32), device=device)
    members_np = np.asarray(designs["members"]).astype(int)
    members = torch.as_tensor(members_np, dtype=torch.long, device=device)
    lengths_np = np.asarray(designs["member_lengths"], dtype=np.float32)
    lengths = torch.as_tensor(lengths_np, dtype=torch.float32, device=device)
    member_layer_np = np.asarray(designs["member_layer"]).astype(int)
    family_code_np = np.asarray(designs["family_code"]).astype(int)
    member_layer = torch.as_tensor(member_layer_np, dtype=torch.long, device=device)
    family_code = torch.as_tensor(family_code_np, dtype=torch.long, device=device)

    a = members[:, 0]
    b = members[:, 1]
    vec = nodes[b] - nodes[a]
    L = torch.linalg.norm(vec, dim=1)
    dirs = vec / torch.clamp(L[:, None], min=EPS)
    abs_dirs = torch.abs(dirs)
    mids = (nodes[a] + nodes[b]) / 2.0
    mids_center = mids - nodes.mean(dim=0, keepdim=True)
    mids_scale = torch.clamp(nodes.std(dim=0, keepdim=True), min=EPS)
    mids_norm = mids_center / mids_scale
    z = nodes[:, 2]
    edge_z_span = torch.abs(nodes[b, 2] - nodes[a, 2])[:, None]
    top_touch = ((torch.isclose(nodes[a, 2], z.max())) | (torch.isclose(nodes[b, 2], z.max()))).float()[:, None]
    bottom_touch = ((torch.isclose(nodes[a, 2], z.min())) | (torch.isclose(nodes[b, 2], z.min()))).float()[:, None]
    radius = torch.linalg.norm(mids_center[:, :2], dim=1, keepdim=True)

    length_feat = torch.log(lengths + EPS)[:, None]
    fam_n = int(family_code_np.max()) + 1
    layer_n = int(member_layer_np.max()) + 1
    fam_oh = torch.nn.functional.one_hot(family_code, num_classes=fam_n).float()
    layer_oh = torch.nn.functional.one_hot(member_layer, num_classes=layer_n).float()

    static_edge = torch.cat([
        length_feat, fam_oh, layer_oh, mids_norm, dirs, abs_dirs,
        edge_z_span, radius, top_touch, bottom_touch
    ], dim=1)

    layer_family_np = member_layer_np * max(1, fam_n) + family_code_np

    return {
        "static_edge": static_edge,
        "lengths": lengths,
        "lengths_np": lengths_np,
        "member_layer_np": member_layer_np,
        "family_code_np": family_code_np,
        "layer_family_np": layer_family_np,
        "n_edges": len(lengths_np),
    }


def group_mean_std_torch(values, group_ids_np):
    """values: (B,E), group ids: (E,) numpy int. Returns mean/std broadcast to edges."""
    B, E = values.shape
    mean = torch.zeros_like(values)
    std = torch.zeros_like(values)
    for g in np.unique(group_ids_np):
        idx_np = np.flatnonzero(group_ids_np == g)
        idx = torch.as_tensor(idx_np, dtype=torch.long, device=values.device)
        vals = values[:, idx]
        mu = vals.mean(dim=1, keepdim=True)
        sig = vals.std(dim=1, keepdim=True, unbiased=False).clamp_min(EPS)
        mean[:, idx] = mu
        std[:, idx] = sig
    return mean, std


def make_features_torch(area_1d, failure_strains, static, norm):
    """Build opt_area_context features for batch of failure fields.

    area_1d: (E,)
    failure_strains: (B,E)
    returns normalized edge features (B,E,F)
    """
    B, E = failure_strains.shape
    area = area_1d[None, :].expand(B, E)
    lengths = static["lengths"]
    volume = area * lengths[None, :]

    static_b = static["static_edge"][None, :, :].expand(B, E, static["static_edge"].shape[1])
    area_feats = torch.stack([
        area,
        torch.log(area.clamp_min(EPS)),
        volume,
        torch.log(volume.clamp_min(EPS)),
    ], dim=-1)
    fs_log = torch.log10(failure_strains.clamp_min(1e-12))[:, :, None]

    layer_mean, layer_std = group_mean_std_torch(area, static["member_layer_np"])
    fam_mean, fam_std = group_mean_std_torch(area, static["family_code_np"])
    lf_mean, lf_std = group_mean_std_torch(area, static["layer_family_np"])

    vol_layer_mean, _ = group_mean_std_torch(volume, static["member_layer_np"])
    vol_fam_mean, _ = group_mean_std_torch(volume, static["family_code_np"])
    vol_lf_mean, _ = group_mean_std_torch(volume, static["layer_family_np"])

    rel_layer = area / layer_mean.clamp_min(EPS)
    rel_fam = area / fam_mean.clamp_min(EPS)
    rel_lf = area / lf_mean.clamp_min(EPS)

    context = torch.stack([
        rel_layer,
        torch.log(rel_layer.clamp_min(EPS)),
        (area - layer_mean) / layer_std.clamp_min(EPS),

        rel_fam,
        torch.log(rel_fam.clamp_min(EPS)),
        (area - fam_mean) / fam_std.clamp_min(EPS),

        rel_lf,
        torch.log(rel_lf.clamp_min(EPS)),
        (area - lf_mean) / lf_std.clamp_min(EPS),

        volume / vol_layer_mean.clamp_min(EPS),
        volume / vol_fam_mean.clamp_min(EPS),
        volume / vol_lf_mean.clamp_min(EPS),
    ], dim=-1)

    edge = torch.cat([static_b, area_feats, fs_log, context], dim=-1)

    x_mu = norm["x_mu"].to(edge.device)
    x_sig = norm["x_sig"].to(edge.device)
    edge_n = (edge - x_mu[None, None, :]) / x_sig[None, None, :]
    return edge_n


def energy_from_pred_curve(pred_curve, norm):
    y_mu = norm["y_mu"].to(pred_curve.device)
    y_sig = norm["y_sig"].to(pred_curve.device)
    curve = pred_curve * y_sig + y_mu
    f = torch.abs(curve)
    disp = DISPLACEMENTS.to(pred_curve.device)
    dx = disp[1:] - disp[:-1]
    energy = (0.5 * (f[:, 1:] + f[:, :-1]) * dx[None, :]).sum(dim=1)
    return energy, curve


def load_model(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    feature_info = ckpt.get("feature_info", {})
    edge_dim = int(feature_info.get("n_edge_features", len(ckpt["normalization"]["x_mu"])))
    hidden = 512
    # Most project models use hidden=512 here; infer from first weight if possible.
    for k, v in ckpt["state_dict"].items():
        if k == "edge_mlp.0.weight":
            hidden = int(v.shape[0])
            break
    model = EdgeSetCurveMLP(edge_in_dim=edge_dim, seq_len=100, hidden=hidden, edge_hidden=hidden)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    norm = {}
    for key in ["x_mu", "x_sig", "y_mu", "y_sig"]:
        norm[key] = torch.as_tensor(ckpt["normalization"][key], dtype=torch.float32, device=device)
    return model, norm, feature_info


def evaluate_area(model, area, failure_fields, static, norm):
    with torch.no_grad():
        x = make_features_torch(area, failure_fields, static, norm)
        pred = model(x)
        energy, curve = energy_from_pred_curve(pred, norm)
    return energy.detach().cpu().numpy(), curve.detach().cpu().numpy()


def optimize_one(start_area_np, model, failure_fields, static, norm, args, device):
    theta_init = inverse_sigmoid_area(start_area_np)
    theta = torch.tensor(theta_init, dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.Adam([theta], lr=args.lr)
    lengths = static["lengths"]
    target_vol = lengths.sum()

    history = []
    best = None

    for step in range(args.steps + 1):
        area = LO + (HI - LO) * torch.sigmoid(theta)
        vol_ratio = (area * lengths).sum() / target_vol

        x = make_features_torch(area, failure_fields, static, norm)
        pred = model(x)
        energy, _ = energy_from_pred_curve(pred, norm)
        mean_e = energy.mean()
        std_e = energy.std(unbiased=False)
        robust_score = mean_e - args.risk_weight * std_e
        vol_pen = args.volume_penalty * (vol_ratio - 1.0) ** 2
        move_pen = args.move_penalty * torch.mean((area - torch.as_tensor(start_area_np, dtype=torch.float32, device=device)) ** 2)
        loss = -robust_score + vol_pen + move_pen

        if step % args.print_every == 0 or step == args.steps:
            rec = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "mean_energy": float(mean_e.detach().cpu()),
                "std_energy": float(std_e.detach().cpu()),
                "robust_score": float(robust_score.detach().cpu()),
                "vol_ratio": float(vol_ratio.detach().cpu()),
                "area_min": float(area.min().detach().cpu()),
                "area_max": float(area.max().detach().cpu()),
            }
            history.append(rec)
            print(json.dumps(rec), flush=True)

        current_score = float(robust_score.detach().cpu() - vol_pen.detach().cpu())
        if best is None or current_score > best["score"]:
            best = {
                "score": current_score,
                "area": area.detach().cpu().numpy().astype(np.float32),
                "step": int(step),
                "mean_energy": float(mean_e.detach().cpu()),
                "std_energy": float(std_e.detach().cpu()),
                "vol_ratio": float(vol_ratio.detach().cpu()),
            }

        if step == args.steps:
            break

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([theta], args.grad_clip)
        opt.step()

    return best, pd.DataFrame(history)


def write_design_artifact(out_dir, source_designs, optimized_areas, rows, lengths):
    out_dir.mkdir(parents=True, exist_ok=True)
    projected = np.stack([project_volume_clip_np(a, lengths) for a in optimized_areas], axis=0)

    save = {k: v for k, v in source_designs.items()}
    save["area_multipliers"] = projected.astype(np.float32)
    save["selected_design_ids"] = np.arange(projected.shape[0], dtype=np.int32)

    setup_path = out_dir / "area_pilot_setup.npz"
    np.savez_compressed(setup_path, **save)

    table = pd.DataFrame(rows)
    table["design_id"] = np.arange(len(rows), dtype=int)
    table["area_min_projected"] = projected.min(axis=1)
    table["area_max_projected"] = projected.max(axis=1)
    table["rel_volume_error_projected"] = (projected @ lengths - np.sum(lengths)) / np.sum(lengths)
    table_path = out_dir / "optimized_area_conditioned_design_table.csv"
    table.to_csv(table_path, index=False)

    artifact_zip = out_dir / "surrogate_optimized_area_designs_artifacts.zip"
    if artifact_zip.exists():
        artifact_zip.unlink()
    with zipfile.ZipFile(artifact_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(setup_path, arcname=setup_path.name)
        z.write(table_path, arcname=table_path.name)

    np.savez_compressed(
        out_dir / "surrogate_optimized_area_multipliers.npz",
        area_multipliers=projected.astype(np.float32),
        preprojection_area_multipliers=np.stack(optimized_areas, axis=0).astype(np.float32),
    )
    return artifact_zip, table_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("outputs/edgeset_force_curve_opt_area_context_fullcombo_h512/model.pt"))
    ap.add_argument("--design-artifact", type=Path, default=Path("outputs/area_paired_v2_fullcombo/nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip"))
    ap.add_argument("--failure-strains", type=Path, default=Path("outputs/area_paired_v2_fullcombo/paired_failure_strains.npz"))
    ap.add_argument("--out", type=Path, default=Path("outputs/surrogate_gradient_opt_opt_area_context"))
    ap.add_argument("--start-design-ids", default="298,288,272,259,265")
    ap.add_argument("--n-failure-samples", type=int, default=32)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.035)
    ap.add_argument("--risk-weight", type=float, default=0.15)
    ap.add_argument("--volume-penalty", type=float, default=1000.0)
    ap.add_argument("--move-penalty", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--print-every", type=int, default=50)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    print("device:", device)

    source_designs, design_member = load_npz_from_zip(args.design_artifact, "area_conditioned_paired_designs.npz")
    failure_pack = np.load(args.failure_strains, allow_pickle=True)
    model, norm, feature_info = load_model(args.model, device)
    static = build_static_tensors(source_designs, device)

    failure_fields, failure_idxs = unique_failure_fields(failure_pack, args.n_failure_samples, args.seed, device)
    print(f"Using {failure_fields.shape[0]} unique failure fields")

    all_areas = np.asarray(source_designs["area_multipliers"], dtype=np.float32)
    start_ids = [int(x) for x in args.start_design_ids.split(",") if x.strip()]
    lengths_np = static["lengths_np"]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    opt_areas = []

    for sid in start_ids:
        print("\n" + "="*72)
        print(f"Optimizing from start design {sid}")
        print("="*72)

        start_area_np = all_areas[sid]
        start_area_t = torch.as_tensor(start_area_np, dtype=torch.float32, device=device)
        start_e, _ = evaluate_area(model, start_area_t, failure_fields, static, norm)
        print(f"start predicted mean={start_e.mean():.6f} std={start_e.std():.6f}")

        best, hist = optimize_one(start_area_np, model, failure_fields, static, norm, args, device)
        hist.to_csv(args.out / f"history_start_{sid:03d}.csv", index=False)

        best_area_projected = project_volume_clip_np(best["area"], lengths_np)
        best_area_t = torch.as_tensor(best_area_projected, dtype=torch.float32, device=device)
        opt_e, _ = evaluate_area(model, best_area_t, failure_fields, static, norm)

        opt_areas.append(best_area_projected)
        rows.append({
            "source_design_id": sid,
            "design_name": f"surrogate_opt_from_{sid:03d}",
            "start_pred_mean_energy": float(start_e.mean()),
            "start_pred_std_energy": float(start_e.std()),
            "start_pred_robust_score": float(start_e.mean() - args.risk_weight * start_e.std()),
            "opt_pred_mean_energy": float(opt_e.mean()),
            "opt_pred_std_energy": float(opt_e.std()),
            "opt_pred_robust_score": float(opt_e.mean() - args.risk_weight * opt_e.std()),
            "pred_mean_gain": float(opt_e.mean() - start_e.mean()),
            "pred_robust_gain": float((opt_e.mean() - args.risk_weight * opt_e.std()) - (start_e.mean() - args.risk_weight * start_e.std())),
            "best_internal_step": int(best["step"]),
            "preprojection_vol_ratio": float(best["vol_ratio"]),
            "area_min": float(best_area_projected.min()),
            "area_max": float(best_area_projected.max()),
        })

    artifact_zip, table_path = write_design_artifact(args.out, source_designs, opt_areas, rows, lengths_np)

    summary = {
        "model": str(args.model),
        "design_artifact": str(args.design_artifact),
        "failure_strains": str(args.failure_strains),
        "start_design_ids": start_ids,
        "n_failure_samples": int(failure_fields.shape[0]),
        "risk_weight": args.risk_weight,
        "volume_penalty": args.volume_penalty,
        "move_penalty": args.move_penalty,
        "steps": args.steps,
        "lr": args.lr,
        "artifact_zip": str(artifact_zip),
        "table": str(table_path),
        "feature_info": feature_info,
    }
    (args.out / "optimization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out / "optimized_design_summary.csv", index=False)

    print("\nDONE")
    print("Optimized design artifact:", artifact_zip)
    print("Summary table:", args.out / "optimized_design_summary.csv")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
