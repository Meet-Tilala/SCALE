#!/usr/bin/env python3
"""
Experiment B -- payment-level simulation of each topology in the Spider OMNeT++ simulator.

For every  topology (from run_structural.py --save-topologies)  x  routing scheme  x  mean channel capacity:
   1. write the Spider topology file + .ned           (to_spider.py)
   2. create (once per N/seed/scale/dag) a workload    (make_workload.py; the SAME file for every method => paired design)
   3. create the .ini with the repo's scripts/create_ini_file.py, then point resultPrefix at our results dir
   4. run ./spiderNet in Cmdenv
   5. sum the per-pair scalars in the .sca  -> success ratio / success volume (counters are reset after each
      recording and only count payments sent inside [transStatStart, transStatEnd], so the sum is the window total)

  python3 run_spider.py --smoke --dry-run            # print what would run
  python3 run_spider.py --smoke --jobs 2             # tiny end-to-end check (minutes)
  python3 run_spider.py --jobs 4                     # full experiment
Re-running skips jobs that already finished (results/spider_runs/<tag>.json).
"""
import argparse, concurrent.futures as cf, glob, json, os, re, shlex, subprocess, sys, time
import numpy as np
import pandas as pd
from sv_growth import graph_from_dict
from to_spider import write_spider_topology, write_ned
from make_workload import make_workload

HERE = os.path.dirname(os.path.abspath(__file__))
SUPPORTED = ["shortestPath", "waterfilling", "lndBaseline", "landmarkRouting"]


def find_binary(repo):
    for p in ["spiderNet", "out/gcc-release/spiderNet", "out/clang-release/spiderNet", "out/gcc-debug/spiderNet"]:
        if os.path.exists(os.path.join(repo, p)):
            return os.path.join(repo, p)
    return None


def summarize_sca(path):
    """Sum the per-(src,dst) scalars written by hostNodeBase (see recordScalar calls in hostNodeBase.cc)."""
    tot = dict(rateCompleted=0.0, amtCompleted=0.0, rateArrived=0.0, amtArrived=0.0,
               rateAttempted=0.0, amtAttempted=0.0, completionTime=0.0)
    with open(path) as f:
        for line in f:
            if not line.startswith("scalar"):
                continue
            try:
                tok = shlex.split(line)
                name, val = tok[-2], float(tok[-1])
            except Exception:
                continue
            if "->" not in name:
                continue
            key = name.split()[0]
            if key in tot:
                tot[key] += val
    return dict(success_ratio=tot["rateCompleted"] / max(tot["rateArrived"], 1.0),
                success_volume=tot["amtCompleted"] / max(tot["amtArrived"], 1e-9),
                completed=tot["rateCompleted"], arrived=tot["rateArrived"],
                avg_completion_time=tot["completionTime"] / max(tot["rateCompleted"], 1.0))


def run_job(job, a, repo, binary):
    t0 = time.time()
    tag, jdir = job["tag"], job["dir"]
    done = os.path.join(a.results, "spider_runs", tag + ".json")
    if os.path.exists(done) and not a.force:
        return json.load(open(done))
    os.makedirs(jdir, exist_ok=True)
    G = graph_from_dict(json.load(open(job["topo_json"])))
    scale_cap = job["mean_cap"] / np.mean([d["cap"] for *_, d in G.edges(data=True)])
    for u, v in G.edges():
        G[u][v]["cap"] *= scale_cap
    topo = os.path.join(jdir, "topo.txt")
    write_spider_topology(G, topo, delay_ms=a.delay)
    net = "net_" + re.sub(r"[^A-Za-z0-9_]", "_", tag)
    write_ned(topo, os.path.join(jdir, net + "_" + job["scheme"] + ".ned"), net, job["scheme"])

    wl = job["workload"]
    if not os.path.exists(wl):
        os.makedirs(os.path.dirname(wl), exist_ok=True)
        make_workload(G.number_of_nodes(), wl, duration=a.duration, scale=a.scale, dag_pct=a.dag_pct,
                      seed=job["seed"] * 1000 + a.workload_seed)

    ini = os.path.join(jdir, "run.ini")
    cmd_ini = [sys.executable, os.path.join(repo, "scripts", "create_ini_file.py"),
               "--network-name", os.path.join(jdir, net), "--topo-filename", topo, "--workload-filename", wl,
               "--ini-filename", ini, "--signals-enabled", "false", "--logging-enabled", "false",
               "--simulation-length", str(a.duration), "--stat-collection-rate", "10",
               "--timeout-clear-rate", "1", "--timeout-enabled", "true", "--routing-scheme", job["scheme"],
               "--demand-scale", str(a.scale), "--transStatStart", str(a.stat_start),
               "--transStatEnd", str(a.stat_end), "--path-choice", "shortest",
               "--balance", str(int(job["mean_cap"])), "--scheduling-algorithm", "LIFO", "--mtu", "1.0"]
    if job["scheme"] != "shortestPath":
        cmd_ini += ["--num-path-choices", str(a.num_paths)]
    env = dict(os.environ, OMNET=os.environ.get("OMNET", "omnetpp-5.6"))      # only needed so scripts/config.py imports
    cmd_run = None
    if a.dry_run:
        return dict(tag=tag, dry_run=True, ini_cmd=" ".join(cmd_ini))
    r = subprocess.run(cmd_ini, cwd=repo, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("create_ini_file.py failed:\n" + r.stderr[-1500:])
    text = open(ini).read()
    cfg = re.search(r"\[Config (\S+)\]", text).group(1)
    prefix = os.path.join(os.path.abspath(a.results), "sca", cfg)
    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    text = re.sub(r'\*\*\.resultPrefix = ".*"', f'**.resultPrefix = "{prefix}"', text)
    open(ini, "w").write(text)

    if a.prepare_only:
        return dict(tag=tag, prepared=True, ini=ini, config=cfg)
    cmd_run = [binary, "-u", "Cmdenv", "-f", ini, "-c", cfg, "-n", f"{repo}:{jdir}"]
    with open(os.path.join(jdir, "run.log"), "w") as lg:
        r = subprocess.run(cmd_run, cwd=repo, stdout=lg, stderr=subprocess.STDOUT, timeout=a.timeout)
    if r.returncode != 0:
        raise RuntimeError(f"spiderNet exit {r.returncode}; see {jdir}/run.log")
    sca = glob.glob(glob.escape(prefix) + "-*.sca")
    if not sca:
        raise RuntimeError(f"no .sca produced for {cfg}; see {jdir}/run.log")
    res = summarize_sca(sca[0])
    res.update({k: job[k] for k in ("tag", "method", "n", "seed", "alpha", "dk", "scheme", "mean_cap")},
               scale=a.scale, dag_pct=a.dag_pct, runtime_s=time.time() - t0, config=cfg)
    json.dump(res, open(done, "w"))
    if not a.keep_vec:
        for f in glob.glob(glob.escape(prefix) + "-*.vec") + glob.glob(glob.escape(prefix) + "-*.vci"):
            os.remove(f)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.abspath(os.path.join(HERE, "..")), help="Spider repo root (contains spiderNet)")
    ap.add_argument("--topo-dir", default="topologies")
    ap.add_argument("--bench", default="bench", help="scratch dir for generated topo/ned/ini/workloads")
    ap.add_argument("--results", default="results")
    ap.add_argument("--methods", nargs="*", help="filter, e.g. pa sv sv_tsr")
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--alpha", type=float, default=None, help="filter S(V) topologies by alpha")
    ap.add_argument("--dk", type=int, default=None, help="filter S(V) topologies by dk")
    ap.add_argument("--schemes", nargs="*", default=["shortestPath", "waterfilling", "lndBaseline"])
    ap.add_argument("--mean-caps", type=float, nargs="*", default=[500, 1000, 2000],
                    help="mean channel capacity values to sweep (the liquidity-scarcity axis)")
    ap.add_argument("--scale", type=int, default=3, help="demand scale (txns/s per unit demand)")
    ap.add_argument("--dag-pct", type=int, default=0, help="percent DAG (imbalanced) demand; 0 = pure circulation")
    ap.add_argument("--duration", type=int, default=1010)
    ap.add_argument("--stat-start", type=int, default=800)
    ap.add_argument("--stat-end", type=int, default=1000)
    ap.add_argument("--num-paths", type=int, default=4)
    ap.add_argument("--delay", type=int, default=30, help="per-hop delay, ms (integer)")
    ap.add_argument("--workload-seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=6 * 3600)
    ap.add_argument("--keep-vec", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="list jobs only, write nothing")
    ap.add_argument("--prepare-only", action="store_true", help="write topo/ned/workload/ini but do not run the simulator")
    ap.add_argument("--smoke", action="store_true", help="tiny run: 1 seed, scale 1, 120 s, one capacity, one scheme")
    a = ap.parse_args()
    if a.smoke:
        a.scale, a.duration, a.stat_start, a.stat_end = 1, 120, 40, 100
        a.mean_caps, a.schemes = a.mean_caps[:1], a.schemes[:1]
        a.seeds = a.seeds or [0]
    for s in a.schemes:
        if s not in SUPPORTED:
            sys.exit(f"scheme {s} not supported by this driver (supported: {SUPPORTED})")
    a.bench, a.results, a.topo_dir = (os.path.abspath(x) for x in (a.bench, a.results, a.topo_dir))   # create_ini_file runs with cwd=repo
    repo = os.path.abspath(a.repo)
    binary = find_binary(repo)
    if binary is None and not (a.dry_run or a.prepare_only):
        sys.exit(f"spiderNet binary not found under {repo}. Build it first (run `make` in the repo; needs OMNeT++).")
    os.makedirs(os.path.join(a.results, "spider_runs"), exist_ok=True)

    jobs = []
    for f in sorted(glob.glob(os.path.join(a.topo_dir, "*.json"))):
        d = json.load(open(f)); meta = d["meta"]
        if a.methods and meta["method"] not in a.methods: continue
        if a.sizes and d["n"] not in a.sizes: continue
        if a.seeds is not None and meta["seed"] not in a.seeds: continue
        if meta["method"] in ("sv", "sv_tsr"):
            if a.alpha is not None and abs(meta["alpha"] - a.alpha) > 1e-9: continue
            if a.dk is not None and meta.get("dk_allowed") != a.dk: continue
        name = os.path.splitext(os.path.basename(f))[0]
        for scheme in a.schemes:
            for cap in a.mean_caps:
                tag = f"{name}_c{int(cap)}_{scheme}"
                jobs.append(dict(tag=tag, topo_json=f, scheme=scheme, mean_cap=cap, method=meta["method"], n=d["n"],
                                 seed=meta["seed"], alpha=meta["alpha"] if meta["method"] in ("sv", "sv_tsr", "greedy_central") else None,
                                 dk=meta.get("dk_allowed"), dir=os.path.join(a.bench, tag),
                                 workload=os.path.join(a.bench, "workloads",
                                                       f"n{d['n']}_s{meta['seed']}_scale{a.scale}_dag{a.dag_pct}_T{a.duration}.txt")))
    print(f"{len(jobs)} jobs; binary={binary}; repo={repo}")
    if not jobs:
        sys.exit("no topologies found -- run run_structural.py --save-topologies topologies first")

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(run_job, j, a, repo, binary): j for j in jobs}
        for i, fu in enumerate(cf.as_completed(futs), 1):
            j = futs[fu]
            try:
                r = fu.result(); rows.append(r)
                print(f"[{i}/{len(jobs)}] {j['tag']}: " + ("dry-run" if r.get("dry_run") else "prepared" if r.get("prepared") else
                      f"success_ratio={r['success_ratio']:.3f} volume={r['success_volume']:.3f}"), flush=True)
            except Exception as e:
                print(f"[{i}/{len(jobs)}] {j['tag']}: FAILED {e}", flush=True)
    if not (a.dry_run or a.prepare_only):
        all_rows = [json.load(open(p)) for p in glob.glob(os.path.join(a.results, "spider_runs", "*.json"))]
        pd.DataFrame(all_rows).to_csv(os.path.join(a.results, "spider.csv"), index=False)
        print("wrote", os.path.join(a.results, "spider.csv"))

if __name__ == "__main__":
    main()
