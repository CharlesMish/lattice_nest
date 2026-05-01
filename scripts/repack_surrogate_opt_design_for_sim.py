#!/usr/bin/env python3
from pathlib import Path
import argparse
import io
import zipfile
import numpy as np
import pandas as pd

def find_member(names, suffix):
    hits = [n for n in names if n.endswith(suffix)]
    if not hits:
        raise FileNotFoundError(f"Could not find a ZIP member ending with {suffix}")
    return sorted(hits, key=len)[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-design-artifact", type=Path,
        default=Path("outputs/area_designs_v2_512/nest_987654_shape15_s012_prune0_area_designs_v2_512_artifacts.zip"))
    ap.add_argument("--optimized-dir", type=Path,
        default=Path("outputs/surrogate_gradient_opt_smoke_298"))
    ap.add_argument("--out", type=Path,
        default=Path("outputs/surrogate_gradient_opt_smoke_298/surrogate_optimized_area_designs_sim_compatible.zip"))
    args = ap.parse_args()

    with zipfile.ZipFile(args.base_design_artifact) as z:
        setup_name = find_member(z.namelist(), "area_pilot_setup.npz")
        raw = z.read(setup_name)
        setup_npz = np.load(io.BytesIO(raw), allow_pickle=True)
        setup = {k: setup_npz[k] for k in setup_npz.files}

    opt_npz_path = args.optimized_dir / "surrogate_optimized_area_multipliers.npz"
    if not opt_npz_path.exists():
        raise FileNotFoundError(opt_npz_path)

    opt = np.load(opt_npz_path, allow_pickle=True)
    area = np.asarray(opt["area_multipliers"], dtype=np.float32)
    setup["area_multipliers"] = area
    setup["selected_design_ids"] = np.arange(area.shape[0], dtype=np.int32)

    table_path = args.optimized_dir / "optimized_area_conditioned_design_table.csv"
    if table_path.exists():
        table = pd.read_csv(table_path)
    else:
        table = pd.DataFrame({
            "design_id": np.arange(area.shape[0], dtype=int),
            "design_name": [f"surrogate_opt_{i:03d}" for i in range(area.shape[0])],
            "category": "surrogate_gradient_opt",
        })

    table["design_id"] = np.arange(len(table), dtype=int)
    if "design_name" not in table.columns:
        table["design_name"] = [f"surrogate_opt_{i:03d}" for i in range(len(table))]
    if "category" not in table.columns:
        table["category"] = "surrogate_gradient_opt"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp_setup = args.out.parent / "_tmp_area_pilot_setup.npz"
    tmp_table = args.out.parent / "_tmp_area_conditioned_design_table.csv"

    np.savez_compressed(tmp_setup, **setup)
    table.to_csv(tmp_table, index=False)

    if args.out.exists():
        args.out.unlink()
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(tmp_setup, arcname="area_pilot_setup.npz")
        z.write(tmp_table, arcname="area_conditioned_design_table.csv")

    tmp_setup.unlink(missing_ok=True)
    tmp_table.unlink(missing_ok=True)

    print("Wrote simulator-compatible design artifact:")
    print(args.out)
    print("n_designs:", area.shape[0], "n_members:", area.shape[1])
    print("has boundary_member_mask:", "boundary_member_mask" in setup)
    print("area min/max:", float(area.min()), float(area.max()))

if __name__ == "__main__":
    main()
