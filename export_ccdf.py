#!/usr/bin/env python3
"""
export_ccdf.py -- write data/ccdf_n300.csv for the pgfplots degree-CCDF figure (the only figure that needs raw data).

  python3 export_ccdf.py --n 300 --alpha 0.5 --dk 15
Reads results/structural.csv (column deg_seq), pools degree sequences over seeds, and writes one column per growth rule:
P(K >= k) for k = 1..max degree; `nan` where the probability is 0 (so the log-axis plot simply ends there).
Upload the CSV to Overleaf as data/ccdf_n300.csv.
"""
import argparse, json, os
import numpy as np, pandas as pd
from analyze import select

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results"); ap.add_argument("--n", type=int, default=300)
ap.add_argument("--alpha", type=float, default=0.5); ap.add_argument("--dk", type=int, default=15)
ap.add_argument("--out", default=None)
a = ap.parse_args()
df = pd.read_csv(os.path.join(a.results, "structural.csv"))
df = select(df[df["n"] == a.n], a.alpha, a.dk)
cols = ["pa", "random", "min_degree", "greedy_central", "sv", "sv_tsr"]
deg = {m: np.concatenate([json.loads(x) for x in df[df["method"] == m]["deg_seq"]]) for m in cols if m in set(df["method"])}
kmax = int(max(d.max() for d in deg.values()))
rows = []
for k in range(1, kmax + 1):
    row = {"k": k}
    for m in cols:
        if m in deg:
            p = float((deg[m] >= k).mean()); row[m] = f"{p:.6g}" if p > 0 else "nan"
        else:
            row[m] = "nan"
    rows.append(row)
out = a.out or os.path.join(a.results, f"ccdf_n{a.n}.csv")
pd.DataFrame(rows).to_csv(out, index=False)
print("wrote", out, f"({len(rows)} rows)")
