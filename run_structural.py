#!/usr/bin/env python3
"""
Experiment A -- structural + economic evaluation of S(V) vs baselines. Needs only networkx/numpy/pandas.

  python3 run_structural.py --preset quick          # ~2 min, sanity check
  python3 run_structural.py --preset paper          # full sweep (N=100,300; 20 seeds)
  python3 run_structural.py --preset scale          # N=1000, 5 seeds (append with --append)

Writes results/structural.csv (one row per method x N x seed x parameter setting) and, with
--save-topologies, the JSON topologies consumed by run_spider.py.
"""
import argparse, itertools, json, os, sys, time
import pandas as pd
from sv_growth import grow, graph_to_dict, METHODS
from structural_metrics import compute_metrics

PRESETS = {  # note: "paper" and "scale" use pivot-sampled centralities (bet_k) for speed
    "quick": dict(sizes=[60, 120], seeds=3, alphas=[0.5], dks=[10], methods=METHODS, m=2),
    "paper": dict(sizes=[100, 300], seeds=20, alphas=[0.2, 0.5, 0.8], dks=[8, 15, 30],
                  methods=METHODS, m=2, bet_k=100),          # ~1-2 h on a laptop
    "scale": dict(sizes=[1000], seeds=5, alphas=[0.5], dks=[15],
                  methods=METHODS, m=2, bet_k=60),           # ~1-2 h; pivot-sampled centralities
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=list(PRESETS), default="quick")
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--seeds", type=int)
    ap.add_argument("--alphas", type=float, nargs="*")
    ap.add_argument("--dks", type=int, nargs="*", help="Delta-k allowed; lambda = (1-alpha)/dk (paper's tuning rule)")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--m", type=int)
    ap.add_argument("--tsr-model", default="load", choices=["capacity", "load"],
                    help="TSR model used by sv_tsr while growing")
    ap.add_argument("--eval-tsr-model", default="load", choices=["capacity", "load"],
                    help="TSR model used to EVALUATE every graph (same for all methods)")
    ap.add_argument("--kappa", type=float, default=0.3)
    ap.add_argument("--cap-dist", default="lognormal", choices=["const", "lognormal", "lnd"])
    ap.add_argument("--lnd-edgelist", default="../lnd_data/lnd_july15_2019_reducedsize.edgelist")
    ap.add_argument("--mean-cap", type=float, default=1000.0)
    ap.add_argument("--bet-k", type=int, default=None, help="sample k pivots for betweenness (speed, N>300)")
    ap.add_argument("--recompute-each", action="store_true", help="re-score after each of the m channels")
    ap.add_argument("--econ-ratio", type=float, default=1.0, help="tau*f/(r*c)")
    ap.add_argument("--out", default="results/structural.csv")
    ap.add_argument("--append", action="store_true", help="append to an existing CSV (e.g. add the scale preset)")
    ap.add_argument("--save-topologies", default=None, help="dir to write JSON topologies (for Spider)")
    ap.add_argument("--save-sizes", type=int, nargs="*", default=[100])
    a = ap.parse_args()

    P = PRESETS[a.preset]
    sizes = a.sizes or P["sizes"]; seeds = a.seeds or P["seeds"]
    alphas = a.alphas or P["alphas"]; dks = a.dks or P["dks"]
    methods = a.methods or P["methods"]; m = a.m or P["m"]
    if a.bet_k is None:
        a.bet_k = P.get("bet_k")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    if a.save_topologies:
        os.makedirs(a.save_topologies, exist_ok=True)

    rows, t0 = [], time.time()
    if a.append and os.path.exists(a.out):
        rows = pd.read_csv(a.out).to_dict("records")
    for n, seed, method in itertools.product(sizes, range(seeds), methods):
        # only the S(V) family depends on (alpha, dk); greedy_central depends on alpha; others run once
        if method in ("sv", "sv_tsr"):
            grid = list(itertools.product(alphas, dks))
        elif method == "greedy_central":
            grid = [(al, None) for al in alphas]
        else:
            grid = [(None, None)]
        for alpha, dk in grid:
            G = grow(method, n, m=m, alpha=(alpha if alpha is not None else 0.5), dk_allowed=dk, seed=seed, cap_dist=a.cap_dist,
                     mean_cap=a.mean_cap, lnd_edgelist=a.lnd_edgelist, tsr_model=a.tsr_model,
                     kappa=a.kappa, bet_k=a.bet_k, tsr_sample_k=a.bet_k, recompute_each=a.recompute_each)
            met = compute_metrics(G, eval_tsr_model=a.eval_tsr_model, kappa=a.kappa, bet_k=a.bet_k,
                                  dk_allowed=dk, econ_ratio=a.econ_ratio, seed=seed)
            met.update(deg_seq=json.dumps(sorted([d for _, d in G.degree()])), method=method, n=n, seed=seed, alpha=alpha, dk=dk, m=m,
                       lam=G.graph["lam"], tsr_grow=a.tsr_model, tsr_eval=a.eval_tsr_model)
            rows.append(met)
            if a.save_topologies and n in a.save_sizes:
                tag = f"{method}_n{n}_a{alpha}_dk{dk}_s{seed}"
                with open(os.path.join(a.save_topologies, tag + ".json"), "w") as f:
                    json.dump(graph_to_dict(G), f)
            print(f"[{time.time()-t0:6.0f}s] {method:15s} n={n} seed={seed} a={alpha} dk={dk}  "
                  f"maxdeg={met['max_deg']} diam={met['diameter']} betgini={met['bet_gini']:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(a.out, index=False, mode="w")
    print("wrote", a.out)

if __name__ == "__main__":
    main()
