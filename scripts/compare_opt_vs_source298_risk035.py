from pathlib import Path
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

opt_zip = Path(
    "outputs/surrogate_gradient_opt_298_move025_risk035_validation_80seeds/"
    "nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
)

source_zip = Path(
    "outputs/area_paired_v2_top_confirm_80seeds/"
    "nest_987654_shape15_s012_prune0_area_conditioned_paired_artifacts.zip"
)

out_csv = Path(
    "outputs/surrogate_gradient_opt_298_move025_risk035_validation_80seeds/"
    "compare_opt_vs_source298.csv"
)

source_design_id = 298

opt = read_run_table(opt_zip)
src = read_run_table(source_zip)

# Optimized artifact has one design, reindexed as design_id 000.
opt_d = opt[opt["design_id"] == 0].copy()

# Original source design 298 from the 80-seed top-confirmation run.
src_d = src[src["design_id"] == source_design_id].copy()

print("Optimized status counts:")
print(opt_d["run_status"].value_counts())

print(f"\nOriginal {source_design_id} status counts:")
print(src_d["run_status"].value_counts())

m = opt_d[["sample_id", "energy", "run_status"]].merge(
    src_d[["sample_id", "energy", "run_status"]],
    on="sample_id",
    suffixes=("_opt", "_src298"),
)

def summarize(df, label):
    df = df.copy()
    df["ratio_opt_vs_src298"] = df["energy_opt"] / df["energy_src298"]

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
        "source 298 energy mean/median/p10/p90:",
        df["energy_src298"].mean(),
        df["energy_src298"].median(),
        df["energy_src298"].quantile(0.10),
        df["energy_src298"].quantile(0.90),
    )
    print(
        "ratio mean/median/p10/p90:",
        df["ratio_opt_vs_src298"].mean(),
        df["ratio_opt_vs_src298"].median(),
        df["ratio_opt_vs_src298"].quantile(0.10),
        df["ratio_opt_vs_src298"].quantile(0.90),
    )
    print(
        "beats source count/fraction:",
        int((df["energy_opt"] > df["energy_src298"]).sum()),
        float((df["energy_opt"] > df["energy_src298"]).mean()),
    )

summarize(m, "all paired rows")

complete = m[
    (m["run_status_opt"] == "complete_physical_curve")
    & (m["run_status_src298"] == "complete_physical_curve")
].copy()

summarize(complete, "complete-vs-complete rows")

out_csv.parent.mkdir(parents=True, exist_ok=True)
m.to_csv(out_csv, index=False)
print("\nwrote", out_csv)
