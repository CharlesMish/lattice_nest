#!/usr/bin/env python3
from pathlib import Path
import argparse
import zipfile, io
import pandas as pd

def read_run_table(zip_or_dir):
    p = Path(zip_or_dir)
    if p.is_dir():
        hits = sorted(p.glob("*run_table.csv"))
        if not hits:
            raise FileNotFoundError(f"No run_table.csv in {p}")
        return pd.read_csv(hits[0])

    with zipfile.ZipFile(p) as z:
        name = next(n for n in z.namelist() if n.endswith("_run_table.csv"))
        return pd.read_csv(io.BytesIO(z.read(name)))

def summarize(df, label, source_design_id):
    df = df.copy()
    ratio_col = f"ratio_opt_vs_src{source_design_id}"
    df[ratio_col] = df["energy_opt"] / df["energy_src"]

    print(f"\n=== {label} ===")
    print("n pairs:", len(df))
    print(
        "optimized energy mean/median/p10/p90:",
        df["energy_opt"].mean(),
        df["energy_opt"].median(),
        df["energy_opt"].quantile(0.10),
        df["energy_opt"].quantile(0.90),
    )
    print(
        f"source {source_design_id} energy mean/median/p10/p90:",
        df["energy_src"].mean(),
        df["energy_src"].median(),
        df["energy_src"].quantile(0.10),
        df["energy_src"].quantile(0.90),
    )
    print(
        "ratio mean/median/p10/p90:",
        df[ratio_col].mean(),
        df[ratio_col].median(),
        df[ratio_col].quantile(0.10),
        df[ratio_col].quantile(0.90),
    )
    print(
        "beats source count/fraction:",
        int((df["energy_opt"] > df["energy_src"]).sum()),
        float((df["energy_opt"] > df["energy_src"]).mean()),
    )

def main():
    ap = argparse.ArgumentParser(
        description="Compare one surrogate-optimized design against its original source design using paired sample_id rows."
    )
    ap.add_argument(
        "--opt-zip",
        default=(
            "outputs/surrogate_gradient_opt_top5_move025_risk035_validation_80seeds/"
            "nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
        ),
        help="Validation artifact ZIP containing optimized designs.",
    )
    ap.add_argument(
        "--source-zip",
        default=(
            "outputs/area_paired_v2_top_confirm_80seeds/"
            "nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
        ),
        help="Top-confirmation artifact ZIP containing original source designs.",
    )
    ap.add_argument(
        "--opt-design-id",
        type=int,
        default=1,
        help="Optimized design id inside the optimized validation artifact. For top5 run: 000=from298, 001=from288.",
    )
    ap.add_argument(
        "--source-design-id",
        type=int,
        default=288,
        help="Original source design id in the top-confirmation artifact.",
    )
    ap.add_argument(
        "--out",
        default=(
            "outputs/surrogate_gradient_opt_top5_move025_risk035_validation_80seeds/"
            "compare_opt001_vs_source288.csv"
        ),
        help="Output CSV path.",
    )
    args = ap.parse_args()

    opt = read_run_table(args.opt_zip)
    src = read_run_table(args.source_zip)

    opt_d = opt[opt["design_id"] == args.opt_design_id].copy()
    src_d = src[src["design_id"] == args.source_design_id].copy()

    if len(opt_d) == 0:
        raise ValueError(f"No optimized rows found for design_id={args.opt_design_id}")
    if len(src_d) == 0:
        raise ValueError(f"No source rows found for design_id={args.source_design_id}")

    print("Optimized status counts:")
    print(opt_d["run_status"].value_counts())

    print(f"\nOriginal {args.source_design_id} status counts:")
    print(src_d["run_status"].value_counts())

    m = opt_d[["sample_id", "energy", "run_status"]].merge(
        src_d[["sample_id", "energy", "run_status"]],
        on="sample_id",
        suffixes=("_opt", "_src"),
    )

    summarize(m, "all paired rows", args.source_design_id)

    complete = m[
        (m["run_status_opt"] == "complete_physical_curve")
        & (m["run_status_src"] == "complete_physical_curve")
    ].copy()

    summarize(complete, "complete-vs-complete rows", args.source_design_id)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    m.to_csv(out, index=False)
    print("\nwrote", out)

if __name__ == "__main__":
    main()
