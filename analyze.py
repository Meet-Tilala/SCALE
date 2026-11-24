#!/usr/bin/env python3
"""
analyze.py -- turn results/*.csv into paper tables (markdown + LaTeX) and figures.

  python3 analyze.py structural --alpha 0.5 --dk 15 --n 300
  python3 analyze.py spider     --alpha 0.5 --dk 15 --n 100
Outputs go to results/tables/ and results/figures/.  Mean +- 95% CI over seeds; paired tests vs. the baseline.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LABEL = {"pa": "Pref. attachment (current LN)", "random": "Random", "greedy_central": "Centrality only (lambda=0)",
         "min_degree": "Min-degree only", "sv": "S(V) hop", "sv_tsr": "S(V)-TSR"}
ORDER = ["pa", "random", "greedy_central", "min_degree", "sv", "sv_tsr"]


def ci95(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    try:
        from scipy import stats
        t = stats.t.ppf(0.975, len(x) - 1)
    except Exception:
        t = 1.96
    return float(t * x.std(ddof=1) / np.sqrt(len(x)))


def paired_p(a, b):
    """two-sided paired test a vs b (Wilcoxon if scipy available)."""
    try:
        from scipy import stats
        return float(stats.wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


def select(df, alpha, dk):
    keep = df["method"].isin(["pa", "random", "min_degree"]) | \
           (df["method"].isin(["sv", "sv_tsr"]) & np.isclose(df["alpha"], alpha) & (df["dk"] == dk)) | \
           ((df["method"] == "greedy_central") & np.isclose(df["alpha"], alpha))
    return df[keep]


def write_tables(tab, name, out):
    os.makedirs(out, exist_ok=True)
    tab.to_csv(os.path.join(out, name + ".csv"))
    with open(os.path.join(out, name + ".md"), "w") as f:
        f.write(tab.to_markdown())
    with open(os.path.join(out, name + ".tex"), "w") as f:
        f.write(tab.to_latex(escape=True))


def structural(a):
    df = pd.read_csv(os.path.join(a.results, "structural.csv"))
    df = df[df["n"] == a.n] if a.n else df
    df = select(df, a.alpha, a.dk)
    cols = ["max_deg", "deg_gini", "diameter", "avg_hops", "bet_gini", "bet_top10pct", "mean_best_path_tsr",
            "breakeven_ratio", "lcc_after_targeted"]
    rows = []
    for m in [m for m in ORDER if m in set(df["method"])]:
        g = df[df["method"] == m]
        row = {"method": LABEL[m], "runs": len(g)}
        for c in cols:
            row[c] = f"{g[c].mean():.3f} ± {ci95(g[c]):.3f}"
        rows.append(row)
    tab = pd.DataFrame(rows).set_index("method")
    out = os.path.join(a.results, "tables"); write_tables(tab, f"structural_n{a.n}", out)
    print(tab.to_string())

    # paired comparison of S(V) variants against the PA baseline (same n, seed)
    base = df[df["method"] == "pa"].set_index(["n", "seed"])
    for m in ["sv", "sv_tsr"]:
        x = df[df["method"] == m].set_index(["n", "seed"])
        j = x.join(base, rsuffix="_pa", how="inner")
        for c in ["max_deg", "bet_gini", "diameter", "mean_best_path_tsr"]:
            d = j[c] - j[c + "_pa"]
            print(f"paired {m} - pa  {c:20s} mean diff {d.mean():+.3f}  p={paired_p(j[c], j[c + '_pa']):.4f}  (n={len(j)})")

    fig = os.path.join(a.results, "figures"); os.makedirs(fig, exist_ok=True)
    # degree CCDF
    plt.figure(figsize=(5, 4))
    for m in [m for m in ORDER if m in set(df["method"])]:
        degs = np.concatenate([json.loads(s) for s in df[df["method"] == m]["deg_seq"]])
        k = np.sort(np.unique(degs)); ccdf = [(degs >= x).mean() for x in k]
        plt.loglog(k, ccdf, marker="o", ms=3, label=LABEL[m])
    plt.xlabel("degree k"); plt.ylabel("P(K >= k)"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(os.path.join(fig, f"degree_ccdf_n{a.n}.pdf")); plt.close()

    # scaling of max degree / diameter with N (uses all N present in the CSV)
    full = select(pd.read_csv(os.path.join(a.results, "structural.csv")), a.alpha, a.dk)
    for col, yl in [("max_deg", "max degree"), ("diameter", "diameter"), ("bet_gini", "betweenness Gini")]:
        plt.figure(figsize=(4.5, 3.6))
        for m in [m for m in ORDER if m in set(full["method"])]:
            g = full[full["method"] == m].groupby("n")[col].agg(["mean", lambda x: ci95(x)])
            plt.errorbar(g.index, g["mean"], yerr=g.iloc[:, 1], marker="o", ms=3, capsize=2, label=LABEL[m])
        plt.xscale("log"); plt.xlabel("N"); plt.ylabel(yl); plt.legend(fontsize=6); plt.tight_layout()
        plt.savefig(os.path.join(fig, f"{col}_vs_N.pdf")); plt.close()


def spider(a):
    df = pd.read_csv(os.path.join(a.results, a.file))
    df = df[df["n"] == a.n] if a.n else df
    df = select(df, a.alpha, a.dk)
    keys = ["n", "seed", "scheme", "mean_cap"]
    out = os.path.join(a.results, "tables"); fig = os.path.join(a.results, "figures"); os.makedirs(fig, exist_ok=True)
    for metric in ["success_ratio", "success_volume"]:
        rows = []
        for (scheme, cap, m), g in df.groupby(["scheme", "mean_cap", "method"]):
            rows.append(dict(scheme=scheme, mean_cap=cap, method=LABEL.get(m, m), mean=g[metric].mean(), ci=ci95(g[metric]), runs=len(g)))
        t = pd.DataFrame(rows)
        t["value"] = t.apply(lambda r: f"{r['mean']:.3f} ± {r['ci']:.3f}", axis=1)
        piv = t.pivot_table(index=["scheme", "mean_cap"], columns="method", values="value", aggfunc="first")
        write_tables(piv, f"spider{a.tag}_{metric}_n{a.n}", out)
        print(f"\n== {metric} ==\n", piv.to_string())
        for scheme in sorted(df["scheme"].unique()):
            plt.figure(figsize=(4.6, 3.6))
            for m in [m for m in ORDER if m in set(df["method"])]:
                g = df[(df["scheme"] == scheme) & (df["method"] == m)].groupby("mean_cap")[metric].agg(["mean", ci95])
                if len(g):
                    plt.errorbar(g.index, g["mean"], yerr=g["ci95"], marker="o", ms=3, capsize=2, label=LABEL[m])
            plt.xscale("log"); plt.xlabel("mean channel capacity"); plt.ylabel(metric.replace("_", " "))
            plt.title(scheme, fontsize=9); plt.legend(fontsize=6); plt.tight_layout()
            plt.savefig(os.path.join(fig, f"{metric}_{scheme}{a.tag}_n{a.n}.pdf")); plt.close()
    # paired differences vs PA
    base = df[df["method"] == "pa"].set_index(keys)
    print("\n== paired difference vs preferential attachment (same workload) ==")
    for m in ["sv", "sv_tsr", "greedy_central", "min_degree", "random"]:
        x = df[df["method"] == m].set_index(keys)
        j = x.join(base, rsuffix="_pa", how="inner")
        for (scheme, cap), g in j.groupby(["scheme", "mean_cap"]):
            d = g["success_ratio"] - g["success_ratio_pa"]
            print(f"{m:15s} {scheme:13s} cap={cap:7.0f}  d(success_ratio)={d.mean():+.4f} ± {ci95(d):.4f}  "
                  f"p={paired_p(g['success_ratio'], g['success_ratio_pa']):.4f}  n={len(g)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", choices=["structural", "spider"])
    ap.add_argument("--results", default="results")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dk", type=int, default=15)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--file", default="spider.csv", help="payment-sim results CSV in --results (pysim.csv for the Python simulator)")
    ap.add_argument("--tag", default="", help="suffix for output file names")
    a = ap.parse_args()
    {"structural": structural, "spider": spider}[a.which](a)
