#!/usr/bin/env python3
from pathlib import Path
import argparse
import sys
import numpy as np
import torch

def energy_np(curve, mask, use_abs=True):
    disp = np.asarray([0.0005 * (i + 1) for i in range(curve.shape[1])], dtype=np.float32)
    out = np.zeros(curve.shape[0], dtype=np.float32)
    fcurve = np.abs(curve) if use_abs else curve
    for i in range(curve.shape[0]):
        valid = mask[i].astype(bool)
        if valid.sum() >= 2:
            out[i] = float(np.trapezoid(fcurve[i, valid], disp[valid]))
        elif valid.sum() == 1:
            out[i] = 0.0
        else:
            out[i] = np.nan
    return out

def compute_norms_local(edge_features, y_curve, y_mask, train_mask):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', type=Path, default=Path('outputs/edgeset_force_curve_opt_area_context_fullcombo_h512/model.pt'))
    ap.add_argument('--artifact-zip', type=Path, default=Path('outputs/area_paired_v2_fullcombo/nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip'))
    ap.add_argument('--failure-strains', type=Path, default=Path('outputs/area_paired_v2_fullcombo/paired_failure_strains.npz'))
    ap.add_argument('--out', type=Path, default=Path('outputs/edgeset_force_curve_opt_area_context_fullcombo_h512/model_with_norm.pt'))
    ap.add_argument('--split-seed', type=int, default=20260428)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    from train_force_curve_edgeset_optfeatures import (
        load_npz_from_zip,
        build_features,
        make_design_splits,
    )

    print('Loading training artifact...')
    designs, _ = load_npz_from_zip(args.artifact_zip, 'area_conditioned_paired_designs.npz')
    targets, _ = load_npz_from_zip(args.artifact_zip, 'area_conditioned_paired_targets.npz')
    failure_pack = np.load(args.failure_strains, allow_pickle=True)

    print('Building opt_area_context features to recompute normalization...')
    data = build_features(designs, targets, failure_pack, feature_set='opt_area_context')
    splits = make_design_splits(data['design_ids'], seed=args.split_seed, n_val=8, n_test=8)
    train_designs = set(splits['train_design_ids'])
    train_mask = np.array([int(d) in train_designs for d in data['design_ids']])

    x_mu, x_sig, y_mu, y_sig, e_mu, e_sig = compute_norms_local(
        data['edge_features'],
        data['y_curve'],
        data['y_mask'],
        train_mask,
    )

    print('Loading checkpoint:', args.model)
    ckpt = torch.load(args.model, map_location='cpu', weights_only=False)
    ckpt['normalization'] = {
        'x_mu': x_mu,
        'x_sig': x_sig,
        'y_mu': y_mu,
        'y_sig': y_sig,
        'energy_train_mean': e_mu,
        'energy_train_std': e_sig,
    }
    ckpt.setdefault('feature_info', data.get('feature_info', {}))
    ckpt['normalization_recomputed_from'] = {
        'artifact_zip': str(args.artifact_zip),
        'failure_strains': str(args.failure_strains),
        'split_seed': args.split_seed,
        'feature_set': 'opt_area_context',
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.out)
    print('Wrote:', args.out)
    print('normalization shapes:', {
        k: tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__
        for k, v in ckpt['normalization'].items()
    })

if __name__ == '__main__':
    main()
