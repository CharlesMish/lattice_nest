#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
runs = [
    ("full_fe_h512", "outputs/edgeset_force_curve_fe_v2_512x10_h512/metrics_summary.csv"),
    ("rank_only_h512", "outputs/edgeset_force_curve_fe_v2compact_rankonly_h512/metrics_summary.csv"),
    ("opt_basic_h512", "outputs/edgeset_force_curve_opt_basic_h512/metrics_summary.csv"),
    ("opt_area_context_h512", "outputs/edgeset_force_curve_opt_area_context_h512/metrics_summary.csv"),
]
rows = []
for name, path in runs:
    p = Path(path)
    if not p.exists():
        print("missing", path)
        continue
    df = pd.read_csv(p)
    for _, r in df.iterrows():
        rows.append({"variant": name, **r.to_dict()})
out = pd.DataFrame(rows)
out.to_csv("outputs/optfeature_compare.csv", index=False)
print(out[out["split"] == "test"].sort_values(["energy_rmse", "force_rmse"]).to_string(index=False))
print("\nwrote outputs/optfeature_compare.csv")
