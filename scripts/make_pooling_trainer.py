#!/usr/bin/env python3
from pathlib import Path

src = Path("scripts/train_force_curve_edgeset_fe.py")
dst = Path("scripts/train_force_curve_edgeset_fe_pooling.py")

if not src.exists():
    raise FileNotFoundError("Expected scripts/train_force_curve_edgeset_fe.py. Copy/run the FE force-curve bundle first.")

s = src.read_text()

start = s.index("class EdgeSetCurveMLP(nn.Module):")
end = s.index("\ndef masked_mse", start)

new_class = """class EdgeSetCurveMLP(nn.Module):
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
"""

s = s[:start] + new_class + s[end:]

old_inst = """model = EdgeSetCurveMLP(edge_in_dim=edge.shape[-1], seq_len=y_curve.shape[1],
                            hidden=args.hidden, edge_hidden=args.hidden, dropout=args.dropout).to(device)"""
new_inst = """model = EdgeSetCurveMLP(edge_in_dim=edge.shape[-1], seq_len=y_curve.shape[1],
                            hidden=args.hidden, edge_hidden=args.hidden, dropout=args.dropout,
                            pooling=args.pooling).to(device)"""
if old_inst not in s:
    raise RuntimeError("Could not find model instantiation block to patch.")
s = s.replace(old_inst, new_inst)

old_arg = 'ap.add_argument("--energy-loss-weight", type=float, default=0.2)\n'
new_arg = old_arg + '    ap.add_argument("--pooling", type=str, default="mean_sum", choices=["mean", "mean_sum", "mean_std", "mean_std_max", "mean_std_max_min", "mean_sum_std", "mean_sum_std_max", "mean_sum_std_max_min"])\n'
if old_arg not in s:
    raise RuntimeError("Could not find energy-loss-weight argparse line.")
s = s.replace(old_arg, new_arg)

old_metrics = '"model": "FeatureEngineeredEdgeSetCurveMLP",\n        "device": str(device),'
new_metrics = '"model": "FeatureEngineeredEdgeSetCurveMLP",\n        "pooling": args.pooling,\n        "device": str(device),'
if old_metrics in s:
    s = s.replace(old_metrics, new_metrics)

s = s.replace(
    'print(f"[edge_curve_fe] epoch={epoch:04d} loss={np.mean(losses):.4e} "',
    'print(f"[edge_curve_fe:{args.pooling}] epoch={epoch:04d} loss={np.mean(losses):.4e} "'
)
s = s.replace(
    'print(f"[edge_curve_fe] early stop at epoch {epoch}", flush=True)',
    'print(f"[edge_curve_fe:{args.pooling}] early stop at epoch {epoch}", flush=True)'
)

dst.write_text(s)
print(f"Wrote {dst}")
