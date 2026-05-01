#!/usr/bin/env python3
"""
Combine broad v2 512x10 area-conditioned dataset with top-confirmation 80-seed data.

Purpose:
  Make a single artifact/failure-strain pair usable by existing surrogate trainers.

Default behavior:
  - keep all broad rows
  - append top-confirmation rows only when (design_id, sample_id) is not already present
    so top samples 0..9 do not duplicate broad 512x10 rows.
  - use the broad design NPZ as the canonical 512-design design table/setup.

Outputs:
  outputs/area_paired_v2_fullcombo/
    nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip
    paired_failure_strains.npz
    combine_summary.json
"""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


RUN_RE = re.compile(r".*design_(\d+)_sample_(\d+)\.npz$")


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


def load_failure_pack(path: Path):
    if not path.exists():
        return None
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


def extract_failure_pack_from_artifact(zip_path: Path, out_path: Path):
    records = []
    failure_strains = []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if RUN_RE.match(n))
        if not names:
            raise RuntimeError(f"No run files matching design_###_sample_###.npz found in {zip_path}")
        for name in names:
            m = RUN_RE.match(name)
            design_id = int(m.group(1))
            sample_id = int(m.group(2))
            with z.open(name) as f:
                d = np.load(io.BytesIO(f.read()), allow_pickle=True)
                fs = np.asarray(d["failure_strains"], dtype=np.float32)
                failure_strains.append(fs)
                failure_seed = -1
                run_status = ""
                energy = np.nan
                peak_force = np.nan
                if "scalar" in d.files:
                    try:
                        scalar = json.loads(str(np.asarray(d["scalar"]).item()))
                        failure_seed = int(scalar.get("failure_seed", -1))
                        run_status = str(scalar.get("run_status", ""))
                        energy = float(scalar.get("energy", np.nan))
                        peak_force = float(scalar.get("peak_force", np.nan))
                    except Exception:
                        pass
                records.append({
                    "design_id": design_id,
                    "sample_id": sample_id,
                    "failure_seed": failure_seed,
                    "run_status": run_status,
                    "energy": energy,
                    "peak_force": peak_force,
                })
    df = pd.DataFrame(records)
    arr = np.stack(failure_strains, axis=0).astype(np.float32)
    np.savez_compressed(
        out_path,
        design_ids=df["design_id"].to_numpy(np.int16),
        sample_ids=df["sample_id"].to_numpy(np.int16),
        failure_seeds=df["failure_seed"].to_numpy(np.int64),
        run_status=df["run_status"].astype(str).to_numpy(dtype=object),
        failure_strains=arr,
    )
    df.to_csv(out_path.with_suffix(".index.csv"), index=False)
    return load_failure_pack(out_path)


def make_keep_mask(design_ids, sample_ids, existing_pairs):
    keep = []
    for d, s in zip(design_ids, sample_ids):
        pair = (int(d), int(s))
        keep.append(pair not in existing_pairs)
    return np.asarray(keep, dtype=bool)


def combine_targets(broad, top, top_keep):
    n_b = len(np.asarray(broad["design_ids"]))
    n_t = len(np.asarray(top["design_ids"]))
    combined = {}
    for k, vb in broad.items():
        if k in top:
            vt = top[k]
            if hasattr(vb, "shape") and hasattr(vt, "shape") and vb.shape[:1] == (n_b,) and vt.shape[:1] == (n_t,):
                combined[k] = np.concatenate([vb, vt[top_keep]], axis=0)
            else:
                combined[k] = vb
        else:
            combined[k] = vb
    return combined


def combine_failures(broad_fp, top_fp, top_keep):
    combined = {}
    n_b = len(np.asarray(broad_fp["design_ids"]))
    n_t = len(np.asarray(top_fp["design_ids"]))
    for k, vb in broad_fp.items():
        if k in top_fp:
            vt = top_fp[k]
            if hasattr(vb, "shape") and hasattr(vt, "shape") and vb.shape[:1] == (n_b,) and vt.shape[:1] == (n_t,):
                combined[k] = np.concatenate([vb, vt[top_keep]], axis=0)
            else:
                combined[k] = vb
        else:
            combined[k] = vb
    return combined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broad-artifact", type=Path, default=Path("outputs/area_paired_v2_512x10/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"))
    ap.add_argument("--broad-failures", type=Path, default=Path("outputs/area_paired_v2_512x10/paired_failure_strains.npz"))
    ap.add_argument("--top-artifact", type=Path, default=Path("outputs/area_paired_v2_top_confirm_80seeds/nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"))
    ap.add_argument("--top-failures", type=Path, default=Path("outputs/area_paired_v2_top_confirm_80seeds/paired_failure_strains.npz"))
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/area_paired_v2_fullcombo"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    designs, design_name = load_npz_from_zip(args.broad_artifact, "area_conditioned_paired_designs.npz")
    broad_targets, broad_targets_name = load_npz_from_zip(args.broad_artifact, "area_conditioned_paired_targets.npz")
    top_targets, top_targets_name = load_npz_from_zip(args.top_artifact, "area_conditioned_paired_targets.npz")

    broad_fp = load_failure_pack(args.broad_failures)
    if broad_fp is None:
        raise FileNotFoundError(f"Missing broad failure pack: {args.broad_failures}")

    top_fp = load_failure_pack(args.top_failures)
    if top_fp is None:
        print(f"Top failure pack not found; extracting from {args.top_artifact}")
        top_fp = extract_failure_pack_from_artifact(args.top_artifact, args.top_failures)

    broad_pairs = set(zip(np.asarray(broad_targets["design_ids"]).astype(int), np.asarray(broad_targets["sample_ids"]).astype(int)))
    top_design = np.asarray(top_targets["design_ids"]).astype(int)
    top_sample = np.asarray(top_targets["sample_ids"]).astype(int)
    top_keep = make_keep_mask(top_design, top_sample, broad_pairs)

    combined_targets = combine_targets(broad_targets, top_targets, top_keep)

    # Align top failure rows to top target ordering if needed, then keep same top_keep.
    # Usually both are sorted identically by design/sample. Verify by pair set/order.
    top_fp_pairs = list(zip(np.asarray(top_fp["design_ids"]).astype(int), np.asarray(top_fp["sample_ids"]).astype(int)))
    top_target_pairs = list(zip(top_design, top_sample))
    if top_fp_pairs != top_target_pairs:
        idx = {p: i for i, p in enumerate(top_fp_pairs)}
        reorder = np.asarray([idx[p] for p in top_target_pairs], dtype=int)
        top_fp = {k: (v[reorder] if hasattr(v, "shape") and v.shape[:1] == (len(reorder),) else v) for k, v in top_fp.items()}

    combined_fp = combine_failures(broad_fp, top_fp, top_keep)

    # Sanity check target/failure row pairing.
    t_pairs = list(zip(np.asarray(combined_targets["design_ids"]).astype(int), np.asarray(combined_targets["sample_ids"]).astype(int)))
    f_pairs = list(zip(np.asarray(combined_fp["design_ids"]).astype(int), np.asarray(combined_fp["sample_ids"]).astype(int)))
    if t_pairs != f_pairs:
        raise RuntimeError("Combined target rows and failure rows do not align by (design_id, sample_id).")

    targets_path = args.out_dir / "nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_targets.npz"
    designs_path = args.out_dir / "nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_designs.npz"
    failures_path = args.out_dir / "paired_failure_strains.npz"
    artifact_zip = args.out_dir / "nest_987654_shape15_s012_prune0_area_paired_v2_fullcombo_artifacts.zip"

    np.savez_compressed(targets_path, **combined_targets)
    np.savez_compressed(designs_path, **designs)
    np.savez_compressed(failures_path, **combined_fp)

    if artifact_zip.exists():
        artifact_zip.unlink()
    with zipfile.ZipFile(artifact_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(designs_path, arcname="nest_987654_shape15_s012_prune0_area_conditioned_paired_designs.npz")
        z.write(targets_path, arcname="nest_987654_shape15_s012_prune0_area_conditioned_paired_targets.npz")

    status = np.asarray(combined_targets.get("run_status", []), dtype=str)
    summary = {
        "broad_artifact": str(args.broad_artifact),
        "top_artifact": str(args.top_artifact),
        "broad_rows": int(len(broad_targets["design_ids"])),
        "top_rows_total": int(len(top_targets["design_ids"])),
        "top_rows_appended": int(top_keep.sum()),
        "duplicates_skipped": int((~top_keep).sum()),
        "combined_rows": int(len(combined_targets["design_ids"])),
        "unique_designs": int(len(set(np.asarray(combined_targets["design_ids"]).astype(int)))),
        "status_counts": {str(k): int(v) for k, v in zip(*np.unique(status, return_counts=True))} if len(status) else {},
        "artifact_zip": str(artifact_zip),
        "failure_pack": str(failures_path),
        "designs_member": design_name,
        "broad_targets_member": broad_targets_name,
        "top_targets_member": top_targets_name,
    }
    (args.out_dir / "combine_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
