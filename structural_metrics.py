"""
structural_metrics.py -- topology + economics metrics for the S(V) experiment (no OMNeT++ needed).

All graphs are evaluated under the SAME TSR model (eval_tsr_model) so that the comparison between
growth methods is fair even though only sv_tsr used TSR while growing.

Metrics map to the paper's claims:
  H1 degree concentration / hard cap  -> max_deg, deg_gini, deg_p99, gamma_hat, frac_over_dk
  H1 efficiency                       -> diameter, avg_hops
  H2 liquidity-aware reliability      -> mean_best_path_tsr  (static proxy for Omega: avg over pairs of exp(-d_TSR))
                                         tsr_avg_dist
  H3 routing-revenue distribution     -> bet_gini, bet_top1pct, bet_top10pct, bet_max_over_mean
  H4 profitability (eqs. 4-6)         -> breakeven_ratio, frac_profitable (at econ_ratio), net_profit
  resilience                          -> lcc_after_targeted, lcc_after_random
"""
import math
import random
import numpy as np
import networkx as nx
from sv_growth import refresh_edge_weights


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    if x.size == 0 or x.sum() <= 0:
        return 0.0
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def top_share(x, frac):
    x = np.sort(np.asarray(x, dtype=float))[::-1]
    k = max(1, int(math.ceil(frac * x.size)))
    return float(x[:k].sum() / x.sum()) if x.sum() > 0 else 0.0


def gamma_mle(degrees, kmin):
    """Continuous-approx discrete power-law MLE (Clauset et al. 2009, eq. 3.7). Rough on small graphs."""
    k = np.array([d for d in degrees if d >= kmin], dtype=float)
    if k.size < 10:
        return float("nan")
    return float(1.0 + k.size / np.sum(np.log(k / (kmin - 0.5))))


def _lcc_fraction_after_removal(G, nodes_to_remove):
    H = G.copy()
    H.remove_nodes_from(nodes_to_remove)
    if H.number_of_nodes() == 0:
        return 0.0
    return max(len(c) for c in nx.connected_components(H)) / float(H.number_of_nodes())


def compute_metrics(G, eval_tsr_model="load", tsr_scale=None, kappa=0.3, bet_k=None,
                    dk_allowed=20, econ_ratio=1.0, seed=0, remove_frac=0.05):
    rng = random.Random(seed)
    n, e = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])
    out = dict(N=n, E=e, avg_deg=float(deg.mean()), max_deg=int(deg.max()), min_deg=int(deg.min()),
               deg_gini=gini(deg), deg_p99=float(np.percentile(deg, 99)),
               frac_over_dk=(float((deg > dk_allowed).mean()) if dk_allowed else float("nan")),
               gamma_hat=gamma_mle(deg, max(2, int(deg.min()))))

    # hop-count efficiency
    out["diameter"] = nx.diameter(G) if nx.is_connected(G) else float("nan")
    out["avg_hops"] = nx.average_shortest_path_length(G) if nx.is_connected(G) else float("nan")

    # betweenness (unweighted) = expected routed share under uniform demand; unnormalised pair counts
    k = bet_k if (bet_k and bet_k < n) else None
    bet = nx.betweenness_centrality(G, k=k, normalized=False, seed=seed)
    b = np.array([bet[v] for v in G])
    out.update(bet_gini=gini(b), bet_top1pct=top_share(b, 0.01), bet_top10pct=top_share(b, 0.10),
               bet_max_over_mean=float(b.max() / b.mean()) if b.mean() > 0 else float("nan"))

    # economics, eqs. (4)-(6), in units of r*c:  Pi_v/(rc) = econ_ratio * Cbet(v) - deg(v)
    pi = econ_ratio * b - deg
    out["frac_profitable"] = float((pi > 0).mean())
    out["net_profit"] = float(pi.sum())
    out["breakeven_ratio"] = float(deg.sum() / b.sum()) if b.sum() > 0 else float("nan")   # eq. (6): tau*f/(r*c) must exceed this
    out["profit_gini_pos"] = gini(np.clip(pi, 0, None))

    # liquidity-aware reliability proxy
    H = G.copy()
    refresh_edge_weights(H, eval_tsr_model, tsr_scale or float(np.mean([d["cap"] for *_, d in H.edges(data=True)])),
                         kappa, sample_k=bet_k, rng=rng)
    if eval_tsr_model != "none":
        tot, cnt = 0.0, 0
        dsum = 0.0
        for s, dist in nx.all_pairs_dijkstra_path_length(H, weight="w"):
            for t, d in dist.items():
                if t != s:
                    tot += math.exp(-d)
                    dsum += d
                    cnt += 1
        out["mean_best_path_tsr"] = tot / cnt
        out["tsr_avg_dist"] = dsum / cnt
        out["mean_edge_tsr"] = float(np.mean([d["tsr"] for *_, d in H.edges(data=True)]))
    else:
        out["mean_best_path_tsr"] = out["tsr_avg_dist"] = out["mean_edge_tsr"] = float("nan")

    # resilience: remove top-degree vs random nodes
    kk = max(1, int(round(remove_frac * n)))
    top = [v for v, _ in sorted(G.degree(), key=lambda x: -x[1])[:kk]]
    out["lcc_after_targeted"] = _lcc_fraction_after_removal(G, top)
    out["lcc_after_random"] = _lcc_fraction_after_removal(G, rng.sample(list(G.nodes()), kk))
    return out
