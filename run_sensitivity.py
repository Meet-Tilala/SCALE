#!/usr/bin/env python3
"""
run_sensitivity.py -- sensitivity of the payment-level result to alpha and dk (N=50, python simulator).

For every (alpha, dk) in the grid it
  1. grows the S(V) and S(V)-TSR topologies (run_structural.py) and, once, the preferential-attachment baseline,
  2. runs the payment simulation on them (run_pysim.py),
  3. writes the paired success-ratio difference vs preferential attachment (same seed, same demand, Wilcoxon),
     one table per routing scheme: results_sens/sensitivity_<scheme>.csv and .tex

  python3 run_sensitivity.py                      # full 3x3 grid, 10 seeds, c = 5 10 20 40, sp1 + ksp4
  python3 run_sensitivity.py --jobs 4             # use 4 processes for the payment simulation
  python3 run_sensitivity.py --summary-only       # only rebuild the tables from results_sens/pysim.csv

Everything goes to topologies_sens/ and results_sens/, so the existing topologies/ and results/ are not touched.
Resumable: finished payment runs are cached in results_sens/pysim_runs/.
"""
import argparse, os, subprocess, sys
import numpy as np
import pandas as pd
from analyze import ci95, paired_p

PY = sys.executable


def run(cmd):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True)


def summary(a):
    df = pd.read_csv(os.path.join(a.results, "pysim.csv"))
    df = df[df["n"] == a.n]
    base = df[df["method"] == "pa"].set_index(["seed", "scheme", "mean_cap"])
    caps = sorted(df["mean_cap"].unique())
    for scheme in sorted(df["scheme"].unique()):
        rows = []
        for method, label in [("sv_tsr", "S($V$)-TSR"), ("sv", "S($V$), hop")]:
            for alpha in a.alphas:
                for dk in a.dks:
                    x = df[(df["method"] == method) & np.isclose(df["alpha"], alpha) & (df["dk"] == dk)]
                    x = x.set_index(["seed", "scheme", "mean_cap"])
                    j = x.join(base, rsuffix="_pa", how="inner")
                    row = dict(method=label, alpha=alpha, dk=dk)
                    for c in caps:
                        g = j[(j.index.get_level_values("scheme") == scheme) & (j.index.get_level_values("mean_cap") == c)]
                        if len(g) < 2:
                            row[f"c={c:g}"] = "--"
                            continue
                        d = g["success_ratio"] - g["success_ratio_pa"]
                        p = paired_p(g["success_ratio"], g["success_ratio_pa"])
                        row[f"c={c:g}"] = f"{d.mean():+.3f}" + ("*" if (p == p and p < 0.05) else "")
                        row[f"c={c:g} ci"] = ci95(d)
                        row[f"c={c:g} p"] = p
                    row["runs"] = len(g)
                    rows.append(row)
        t = pd.DataFrame(rows)
        t.to_csv(os.path.join(a.results, f"sensitivity_{scheme}.csv"), index=False)
        show = t[["method", "alpha", "dk"] + [f"c={c:g}" for c in caps]]
        print(f"\n== {scheme}: success-ratio difference vs preferential attachment (* = Wilcoxon p<0.05, {int(t['runs'].max())} seeds) ==")
        print(show.to_string(index=False))
        L = [r"\begin{tabular}{llc" + "c" * len(caps) + "}", r"\toprule",
             r"Growth rule & $\alpha$ & $\Delta k$ & " + " & ".join(f"$c{{=}}{c:g}$" for c in caps) + r" \\", r"\midrule"]
        for _, r in show.iterrows():
            cells = [str(r[f"c={c:g}"]).replace("*", "$^{*}$") for c in caps]
            L.append(f"{r['method']} & {r['alpha']:g} & {int(r['dk'])} & " + " & ".join(cells) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(a.results, f"sensitivity_{scheme}.tex"), "w").write("\n".join(L) + "\n")
    print("\nwrote", os.path.join(a.results, "sensitivity_<scheme>.csv/.tex"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.2, 0.5, 0.8])
    ap.add_argument("--dks", type=int, nargs="*", default=[8, 15, 30])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--mean-caps", type=float, nargs="*", default=[5, 10, 20, 40])
    ap.add_argument("--schemes", nargs="*", default=["sp1", "ksp4"])
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--topo-dir", default="topologies_sens")
    ap.add_argument("--results", default="results_sens")
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    if not a.summary_only:
        base = [PY, "run_structural.py", "--preset", "quick", "--sizes", a.n, "--seeds", a.seeds,
                "--save-topologies", a.topo_dir, "--save-sizes", a.n]
        run(base + ["--methods", "pa", "--out", os.path.join(a.results, "structural_pa.csv")])
        for alpha in a.alphas:
            for dk in a.dks:
                run(base + ["--methods", "sv", "sv_tsr", "--alphas", alpha, "--dks", dk,
                            "--out", os.path.join(a.results, f"structural_a{alpha:g}_dk{dk}.csv")])
        for alpha in a.alphas:
            for dk in a.dks:
                run([PY, "run_pysim.py", "--topo-dir", a.topo_dir, "--results", a.results, "--alpha", alpha, "--dk", dk,
                     "--methods", "pa", "sv", "sv_tsr", "--seeds"] + list(range(a.seeds)) +
                    ["--schemes"] + a.schemes + ["--mean-caps"] + a.mean_caps + ["--jobs", a.jobs])
    summary(a)


if __name__ == "__main__":
    main()
