"""Sanity tests -- run `python3 test_sv.py` before trusting any result."""
import os, tempfile
import numpy as np, networkx as nx
from sv_growth import grow, s_scores, composite_centrality, METHODS
from structural_metrics import gini, compute_metrics
from to_spider import write_spider_topology, write_ned
from run_spider import summarize_sca

def test_penalty_extremes():
    G = nx.star_graph(6)                       # node 0 = hub
    for u, v in G.edges(): G[u][v]["cap"] = 100.0
    sc0 = s_scores(G, 0.5, 0.0, False)
    assert max(sc0, key=sc0.get) == 0, "lambda=0 must pick the hub"
    sc = s_scores(G, 0.5, 10.0, False)
    assert max(sc, key=sc.get) != 0, "huge lambda must avoid the hub"

def test_equal_capital_across_methods():
    Es, caps = set(), []
    for m in METHODS:
        G = grow(m, 40, m=2, dk_allowed=8, seed=3, tsr_model="capacity")
        Es.add(G.number_of_edges()); caps.append(sum(d["cap"] for *_, d in G.edges(data=True)))
    assert len(Es) == 1, Es
    assert np.allclose(caps, caps[0], rtol=1e-9), caps

def test_same_seed_same_graph():
    a = grow("sv_tsr", 30, seed=5, dk_allowed=8); b = grow("sv_tsr", 30, seed=5, dk_allowed=8)
    assert sorted(a.edges()) == sorted(b.edges())

def test_gini():
    assert abs(gini([1, 1, 1, 1])) < 1e-12 and gini([0, 0, 0, 1]) > 0.7

def test_connected_and_metrics():
    G = grow("pa", 50, seed=1)
    assert nx.is_connected(G)
    m = compute_metrics(G, dk_allowed=10)
    assert m["N"] == 50 and m["diameter"] >= 2

def test_spider_files_and_sca_parser():
    G = grow("sv", 20, seed=2, dk_allowed=8)
    with tempfile.TemporaryDirectory() as d:
        t = os.path.join(d, "t.txt"); write_spider_topology(G, t); write_ned(t, os.path.join(d, "n.ned"), "net", "waterfilling")
        lines = open(t).read().splitlines()
        assert lines[0].split()[0].endswith("r") and len(lines) == 1 + G.number_of_edges() + 20
        assert all(x.isdigit() for l in lines[1:] for x in l.split()[2:4])        # integer delays (C++ stoi)
        sca = os.path.join(d, "x.sca")
        open(sca, "w").write('run x\nscalar net.host[0] "rateCompleted 0 -> 1" 8\nscalar net.host[0] "rateArrived 0 -> 1" 10\n'
                             'scalar net.host[0] "amtCompleted 0 -> 1" 4.5\nscalar net.host[0] "amtArrived  0 -> 1" 9\n'
                             'scalar net.host[0] "time" 100\nscalar net.host[0] "rateCompleted 0 -> 1" 2\n'
                             'scalar net.host[0] "rateArrived 0 -> 1" 2\n')
        r = summarize_sca(sca)
        assert abs(r["success_ratio"] - 10 / 12) < 1e-9 and abs(r["success_volume"] - 0.5) < 1e-9

if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f(); print("ok", k)
