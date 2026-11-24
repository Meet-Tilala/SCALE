#!/usr/bin/env python3
"""
Experiment B (no-OMNeT++ version): payment-level simulation with payment_sim.py.
Reads topologies/*.json (made by  ./run_all.sh topos), writes results/pysim.csv.

  python3 run_pysim.py --smoke                  # seconds
  python3 run_pysim.py                          # full N=50 experiment
  python3 run_pysim.py --jobs 4 --mean-caps 5 10 20 40
Resumable: finished rows are kept in results/pysim_runs/.
"""
import argparse, concurrent.futures as cf, glob, json, os, time
import pandas as pd
from sv_growth import graph_from_dict
from payment_sim import workload_arrays, simulate

SCHEMES = {"sp1": 1, "ksp4": 4}


def one(job, a):
    done = os.path.join(a.results, "pysim_runs", job["tag"] + ".json")
    if os.path.exists(done) and not a.force:
        return json.load(open(done))
    d = json.load(open(job["topo"]))
    G = graph_from_dict(d)
    wl = workload_arrays(G.number_of_nodes(), a.duration, a.scale, a.dag_pct, job["seed"] * 1000 + a.workload_seed)
    t0 = time.time()
    r = simulate(G, wl, job["mean_cap"], k=SCHEMES[job["scheme"]], hold=a.hold, stat_start=a.stat_start, stat_end=a.duration)
    r.update({k: job[k] for k in ("tag", "method", "n", "seed", "alpha", "dk", "scheme", "mean_cap")},
             scale=a.scale, dag_pct=a.dag_pct, hold=a.hold, runtime_s=time.time() - t0)
    json.dump(r, open(done, "w"))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topo-dir", default="topologies")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dk", type=int, default=15)
    ap.add_argument("--schemes", nargs="*", default=["sp1", "ksp4"], choices=list(SCHEMES))
    ap.add_argument("--mean-caps", type=float, nargs="*", default=[5, 10, 20, 40],
                    help="mean channel capacity in payment-size units (mean payment ~0.3): the liquidity-scarcity axis")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--dag-pct", type=int, default=0)
    ap.add_argument("--duration", type=int, default=400)
    ap.add_argument("--stat-start", type=int, default=100, help="warm-up before this time is excluded")
    ap.add_argument("--hold", type=float, default=1.0, help="seconds funds stay locked in flight (0 = instant)")
    ap.add_argument("--workload-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.duration, a.stat_start, a.mean_caps, a.schemes, a.seeds = 120, 30, a.mean_caps[:2], ["ksp4"], a.seeds or [0]
    os.makedirs(os.path.join(a.results, "pysim_runs"), exist_ok=True)
    jobs = []
    for f in sorted(glob.glob(os.path.join(a.topo_dir, "*.json"))):
        d = json.load(open(f)); m = d["meta"]
        if a.methods and m["method"] not in a.methods: continue
        if a.seeds is not None and m["seed"] not in a.seeds: continue
        if m["method"] in ("sv", "sv_tsr"):
            if abs(m["alpha"] - a.alpha) > 1e-9 or m.get("dk_allowed") != a.dk: continue
        name = os.path.splitext(os.path.basename(f))[0]
        for sc in a.schemes:
            for cap in a.mean_caps:
                jobs.append(dict(tag=f"{name}_c{cap:g}_{sc}", topo=f, scheme=sc, mean_cap=cap, method=m["method"], n=d["n"],
                                 seed=m["seed"], alpha=m["alpha"] if m["method"] in ("sv", "sv_tsr", "greedy_central") else None,
                                 dk=m.get("dk_allowed")))
    print(len(jobs), "runs", flush=True)
    with cf.ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(one, j, a): j for j in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            r = fu.result()
            print(f"[{i}/{len(jobs)}] {r['tag']}: ratio={r['success_ratio']:.3f} vol={r['success_volume']:.3f} ({r['runtime_s']:.1f}s)", flush=True)
    rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.results, "pysim_runs", "*.json"))]
    pd.DataFrame(rows).to_csv(os.path.join(a.results, "pysim.csv"), index=False)
    print("wrote", os.path.join(a.results, "pysim.csv"))

if __name__ == "__main__":
    main()
