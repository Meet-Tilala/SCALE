#!/usr/bin/env python3
"""
make_paper_tables.py -- compact, IEEE-column-friendly LaTeX tables (booktabs) from results/*.csv.

  python3 make_paper_tables.py --n 100 --alpha 0.5 --dk 15            # structural table for N=100
  python3 make_paper_tables.py --n 300 --alpha 0.5 --dk 15            # structural table for N=300
  python3 make_paper_tables.py --pysim-n 50 --alpha 0.5 --dk 15       # payment tables from pysim.csv

Writes results/tables/paper_structural_n<N>.tex, paper_pysim_success_ratio.tex, paper_pysim_paired.tex
Each file is a bare tabular: wrap it in \\begin{table*} ... \\resizebox{\\textwidth}{!}{\\input{...}} (Evaluation.tex does this).
"""
import argparse, os
import numpy as np
import pandas as pd
from analyze import ci95, paired_p, select

TEX = {"pa": r"Pref.\ attachment (LN)", "random": "Random", "greedy_central": r"Centrality only ($\lambda{=}0$)",
       "min_degree": "Min-degree only", "sv": r"S($V$), hop distance", "sv_tsr": r"S($V$)-TSR"}
ORDER = ["pa", "random", "min_degree", "greedy_central", "sv", "sv_tsr"]
STRUCT_COLS = [("max_deg", "Max deg.", 1), ("deg_gini", "Deg. Gini", 2), ("diameter", "Diam.", 1),
               ("bet_gini", "Betw. Gini", 2), ("bet_top10pct", "Top-10\\% betw.", 2),
               ("mean_best_path_tsr", r"Path TSR", 2), ("lcc_after_targeted", "LCC after attack", 2)]


def pm(x, d):
    return f"{np.nanmean(x):.{d}f}$\\pm${ci95(x):.{d}f}"


def structural(a):
    df = pd.read_csv(os.path.join(a.results, "structural.csv"))
    df = select(df[df["n"] == a.n], a.alpha, a.dk)
    lines = [r"\begin{tabular}{l" + "c" * len(STRUCT_COLS) + "}", r"\toprule",
             "Growth rule & " + " & ".join(h for _, h, _ in STRUCT_COLS) + r" \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(df["method"])]:
        g = df[df["method"] == m]
        lines.append(TEX[m] + " & " + " & ".join(pm(g[c], d) for c, _, d in STRUCT_COLS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    out = os.path.join(a.results, "tables", f"paper_structural_n{a.n}.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write("\n".join(lines) + "\n")
    print("wrote", out, f"(runs per method: {df.groupby('method').size().to_dict()})")


def pysim(a):
    df = pd.read_csv(os.path.join(a.results, "pysim.csv"))
    df = select(df[df["n"] == a.pysim_n], a.alpha, a.dk)
    schemes = [s for s in ["ksp4", "sp1"] if s in set(df["scheme"])]
    caps = sorted(df["mean_cap"].unique())
    # (1) success ratio, mean +- CI
    head = r"\begin{tabular}{l" + ("c" * len(caps)) * len(schemes) + "}"
    L = [head, r"\toprule", "Growth rule & " + " & ".join(
        rf"\multicolumn{{{len(caps)}}}{{c}}{{{s}}}" for s in schemes) + r" \\"]
    L.append("& " + " & ".join(f"$c{{=}}{c:g}$" for _ in schemes for c in caps) + r" \\")
    L.append(r"\midrule")
    for m in [m for m in ORDER if m in set(df["method"])]:
        cells = []
        for s in schemes:
            for c in caps:
                g = df[(df["method"] == m) & (df["scheme"] == s) & (df["mean_cap"] == c)]["success_ratio"]
                cells.append(pm(g, 3) if len(g) else "--")
        L.append(TEX[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    p1 = os.path.join(a.results, "tables", "paper_pysim_success_ratio.tex")
    os.makedirs(os.path.dirname(p1), exist_ok=True)
    open(p1, "w").write("\n".join(L) + "\n")

    # (2) paired difference vs preferential attachment (same topology seed, same demand), one scheme
    s = a.scheme if a.scheme in schemes else schemes[0]
    keys = ["seed", "scheme", "mean_cap"]
    base = df[df["method"] == "pa"].set_index(keys)
    L = [r"\begin{tabular}{l" + "c" * len(caps) + "}", r"\toprule",
         "Growth rule & " + " & ".join(f"$c{{=}}{c:g}$" for c in caps) + r" \\", r"\midrule"]
    for m in [m for m in ORDER if m in set(df["method"]) and m != "pa"]:
        x = df[df["method"] == m].set_index(keys)
        j = x.join(base, rsuffix="_pa", how="inner")
        cells = []
        for c in caps:
            g = j[(j.index.get_level_values("scheme") == s) & (j.index.get_level_values("mean_cap") == c)]
            if len(g) < 2:
                cells.append("--"); continue
            d = g["success_ratio"] - g["success_ratio_pa"]
            p = paired_p(g["success_ratio"], g["success_ratio_pa"])
            star = "$^{*}$" if (p == p and p < 0.05) else ""
            cells.append(f"{d.mean():+.3f}{star}")
        L.append(TEX[m] + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    p2 = os.path.join(a.results, "tables", "paper_pysim_paired.tex")
    open(p2, "w").write("\n".join(L) + "\n")
    print("wrote", p1, "and", p2, f"(paired table uses scheme={s}; * = Wilcoxon p<0.05 vs pref. attachment)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--n", type=int, default=None, help="N for the structural table")
    ap.add_argument("--pysim-n", type=int, default=None, help="N for the payment-simulation tables")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dk", type=int, default=15)
    ap.add_argument("--scheme", default="ksp4")
    a = ap.parse_args()
    if a.n:
        structural(a)
    if a.pysim_n:
        pysim(a)
    if not (a.n or a.pysim_n):
        ap.error("give --n and/or --pysim-n")
