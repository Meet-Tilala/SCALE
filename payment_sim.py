"""
payment_sim.py -- pure-Python payment-channel network simulator (no OMNeT++).  Replaces Experiment B's simulator.

Model (state it in the paper):
  * each channel has capacity c, split 50/50 into two directional balances (as in Spider's topology files)
  * a payment of size x from s to t is tried on up to k shortest (hop-count) simple paths, in order; the first path
    whose every hop has balance >= x is used (k=1 -> 'sp1' single shortest path; k=4 -> 'ksp4', LND-like retry)
  * funds are locked on the way forward and credited to the reverse direction after `hold` seconds
    (hold=0 -> instant settlement). Failed payments are dropped (no queueing, no splitting -- unlike Spider).
  * success ratio / volume are measured only for payments arriving in [stat_start, stat_end] (warm-up excluded)
  * demand: same circulation generator and size distribution as make_workload.py; the same demand is used for every
    topology of a given (N, seed), so comparisons are paired.
"""
import heapq
import itertools
import numpy as np
import networkx as nx
from make_workload import circ_demand, dag_demand, LOG_NORMAL_MEAN, LOG_NORMAL_SCALE, MIN_TXN, MAX_TXN


def workload_arrays(n, duration, scale, dag_pct, seed, mean_rate=10, std_dev=2):
    rng = np.random.default_rng(seed)
    d = dag_pct / 100.0
    D = (1 - d) * circ_demand(n, rng, mean_rate, std_dev) + d * dag_demand(n, rng, mean_rate)
    s_idx, r_idx = np.nonzero(D * scale)
    counts = rng.poisson(D[s_idx, r_idx] * scale * duration)
    tot = int(counts.sum())
    src, dst = np.repeat(s_idx, counts), np.repeat(r_idx, counts)
    t = rng.uniform(0, duration, size=tot)
    sz = np.clip(np.round(10 ** rng.normal(LOG_NORMAL_MEAN, LOG_NORMAL_SCALE, size=tot), 1), MIN_TXN, MAX_TXN)
    o = np.argsort(t)
    return t[o], src[o], dst[o], sz[o]


def simulate(G, wl, mean_cap, k=4, hold=1.0, stat_start=None, stat_end=None):
    """G: nx.Graph with nodes 0..N-1 and edge attr 'cap' (relative sizes are kept; rescaled to mean_cap)."""
    t, src, dst, sz = wl
    n = G.number_of_nodes()
    f = mean_cap / np.mean([d["cap"] for *_, d in G.edges(data=True)])
    bal = {}
    for u, v, d in G.edges(data=True):
        bal[(u, v)] = bal[(v, u)] = d["cap"] * f / 2.0
    adj = G
    cache = {}

    def paths(s, r):
        key = (s, r)
        p = cache.get(key)
        if p is None:
            p = list(itertools.islice(nx.shortest_simple_paths(adj, s, r), k))
            cache[key] = p
        return p

    t0 = stat_start if stat_start is not None else 0.0
    t1 = stat_end if stat_end is not None else float(t[-1]) + 1
    pending = []                      # heap of (release_time, counter, path, amount)
    cnt = 0
    arrived = completed = 0
    amt_arr = amt_ok = 0.0
    hop_sum = 0
    for i in range(len(t)):
        now = float(t[i])
        while pending and pending[0][0] <= now:
            _, _, p, a = heapq.heappop(pending)
            for u, v in zip(p[:-1], p[1:]):
                bal[(v, u)] += a
        x = float(sz[i])
        measured = t0 <= now <= t1
        if measured:
            arrived += 1
            amt_arr += x
        for p in paths(int(src[i]), int(dst[i])):
            hops = list(zip(p[:-1], p[1:]))
            if all(bal[h] >= x for h in hops):
                for h in hops:
                    bal[h] -= x
                if hold > 0:
                    cnt += 1
                    heapq.heappush(pending, (now + hold, cnt, p, x))
                else:
                    for u, v in hops:
                        bal[(v, u)] += x
                if measured:
                    completed += 1
                    amt_ok += x
                    hop_sum += len(hops)
                break
    return dict(success_ratio=completed / max(arrived, 1), success_volume=amt_ok / max(amt_arr, 1e-9),
                completed=completed, arrived=arrived, avg_hops_used=hop_sum / max(completed, 1))
