#!/usr/bin/env python3
from pathlib import Path
import argparse
import pandas as pd
import json

ap = argparse.ArgumentParser()
ap.add_argument("--base-out", required=True, type=Path)
args = ap.parse_args()

rows = []
for p in sorted(args.base_out.glob("*/metrics_summary.csv")):
    pooling = p.parent.name
    df = pd.read_csv(p)
    test = df[df["split"] == "test"].iloc[0].to_dict()
    val = df[df["split"] == "val"].iloc[0].to_dict()
    metrics_json = p.parent / "metrics.json"
    meta = {}
    if metrics_json.exists():
        meta = json.loads(metrics_json.read_text())
    rows.append({
        "pooling": meta.get("pooling", pooling),
        "out_dir": str(p.parent),
        "val_force_rmse": val.get("force_rmse"),
        "val_force_flat_r2": val.get("force_flat_r2"),
        "val_force_mean_per_step_r2": val.get("force_mean_per_step_r2"),
        "val_energy_rmse": val.get("energy_rmse"),
        "val_energy_r2": val.get("energy_r2"),
        "test_force_rmse": test.get("force_rmse"),
        "test_force_flat_r2": test.get("force_flat_r2"),
        "test_force_mean_per_step_r2": test.get("force_mean_per_step_r2"),
        "test_energy_rmse": test.get("energy_rmse"),
        "test_energy_r2": test.get("energy_r2"),
    })

out = pd.DataFrame(rows)
if len(out):
    out = out.sort_values(["test_energy_rmse", "test_force_rmse"], ascending=True)
    out.to_csv(args.base_out / "pooling_ablation_summary.csv", index=False)
    print(out.to_string(index=False))
    print("\nWrote", args.base_out / "pooling_ablation_summary.csv")
else:
    print("No metrics_summary.csv files found under", args.base_out)
