"""
make_workload.py -- Gurobi-free workload generator in the repo's workload format.

Line format (hostInitialize.cc::generateTransUnitList):  <amount> <time_s> <sender_host> <receiver_host> 0 <timeout_s>
Demand follows scripts/create_workload.py: a circulation (sum of `mean_rate` random permutation matrices
plus `std_dev` random cycles) optionally mixed with a DAG component; arrivals per ordered pair are Poisson
with rate = demand * scale; sizes are the repo's log-normal (10**N(-0.6152, 0.7310), clipped to [0.1, 10]).
The SAME workload file is reused for every topology of a given (N, seed) => paired comparison.
"""
import numpy as np

LOG_NORMAL_MEAN, LOG_NORMAL_SCALE = -0.6152, 0.7310      # scripts/config.py
MIN_TXN, MAX_TXN = 0.1, 10.0


def circ_demand(n, rng, mean_rate=10, std_dev=2):
    D = np.zeros((n, n))
    for _ in range(mean_rate):
        perm = rng.permutation(n)
        D[np.arange(n), perm] += 1
    for _ in range(std_dev):
        k = int(rng.integers(2, n + 1))
        cyc = rng.choice(n, size=k, replace=False)
        for a, b in zip(cyc, np.roll(cyc, -1)):
            D[a, b] += 1
    np.fill_diagonal(D, 0)
    return D


def dag_demand(n, rng, mean_rate=10):
    D = np.zeros((n, n))
    order = rng.permutation(n)
    for i, s in enumerate(order[:-1]):
        later = order[i + 1:]
        for r in rng.choice(later, size=mean_rate, replace=True):
            D[s, r] += 1
    return D


def make_workload(n, path, duration=1010, scale=3, dag_pct=0, seed=0, timeout=5, size_mode="lognormal",
                  mean_rate=10, std_dev=2):
    rng = np.random.default_rng(seed)
    d = dag_pct / 100.0
    D = (1 - d) * circ_demand(n, rng, mean_rate, std_dev) + d * dag_demand(n, rng, mean_rate)
    rate = D * scale
    s_idx, r_idx = np.nonzero(rate)
    lam = rate[s_idx, r_idx] * duration
    counts = rng.poisson(lam)
    total = int(counts.sum())
    src = np.repeat(s_idx, counts)
    dst = np.repeat(r_idx, counts)
    t = rng.uniform(0, duration, size=total)
    if size_mode == "lognormal":
        sz = np.round(10 ** rng.normal(LOG_NORMAL_MEAN, LOG_NORMAL_SCALE, size=total), 1)
        sz = np.clip(sz, MIN_TXN, MAX_TXN)
    else:
        sz = np.ones(total)
    o = np.argsort(t)
    with open(path, "w") as f:
        for i in o:
            f.write(f"{sz[i]} {t[i]:.6f} {src[i]} {dst[i]} 0 {int(timeout)}\n")
    return total, float(sz.sum())
