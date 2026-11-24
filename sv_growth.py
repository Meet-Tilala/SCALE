"""
sv_growth.py -- network-growth models for the S(V) experiment.

Implements the objective from the paper
    S(V) = C(V) - lambda * (deg(V) - deg_min),   C = alpha*Closeness_N + (1-alpha)*Betweenness_N
plus the baselines it is compared against.

Methods
  sv              S(V) with hop-count distances (TSR ignored)
  sv_tsr          S(V) with liquidity-aware distances, w(e) = -log(TSR_e)   <- the paper's full proposal
  pa              linear preferential attachment (P ~ degree)               <- "current LN" baseline
  random          uniform attachment
  greedy_central  centrality only (lambda = 0)                              <- ablation: is the penalty needed?
  min_degree      always join the lowest-degree node(s)                     <- ablation: is centrality needed?

Every method starts from the same seed graph, gives each joiner the same number of channels m,
and draws channel capacities from the same distribution, so edges and total locked capital are
identical across methods (the paper's K_net,prop = K_net,current assumption).

Modelling assumptions you must state in the paper (TSR is not directly observable at join time):
  tsr_model = none      TSR_e = 1 (hop-count only)
  tsr_model = capacity  TSR_e = 1 - exp(-(cap_e/2) / tsr_scale)           (bigger channel -> more reliable)
  tsr_model = load      TSR_e = exp(-rho_e), rho_e = kappa*(ebc_e/mean ebc)*(mean cap/cap_e)
                        (channels carrying much more than their capacity share deplete -> lower TSR)
"""
import math
import random
import numpy as np
import networkx as nx

METHODS = ["sv", "sv_tsr", "pa", "random", "greedy_central", "min_degree"]
TSR_MODELS = ["none", "capacity", "load"]
EPS = 1e-6


# ----------------------------------------------------------------------------- capacities
def make_capacity_sampler(dist, mean_cap, rng, lnd_edgelist=None):
    """Return a zero-arg function that draws one channel capacity (mean ~= mean_cap)."""
    if dist == "const":
        return lambda: float(mean_cap)
    if dist == "lognormal":
        sigma = 1.0
        mu = math.log(mean_cap) - sigma ** 2 / 2.0
        return lambda: float(max(1.0, rng.lognormvariate(mu, sigma)))
    if dist == "lnd":
        if lnd_edgelist is None:
            raise ValueError("--cap-dist lnd needs --lnd-edgelist")
        G = nx.read_edgelist(lnd_edgelist, nodetype=str)
        caps = np.array([d["capacity"] for _, _, d in G.edges(data=True)], dtype=float)
        caps = caps * (mean_cap / caps.mean())
        return lambda: float(max(1.0, rng.choice(list(caps))))
    raise ValueError(dist)


# ----------------------------------------------------------------------------- TSR models
def refresh_edge_weights(G, tsr_model, tsr_scale, kappa, tsr_file=None, sample_k=None, rng=None):
    """Set G[u][v]['tsr'] and ['w'] (= -log tsr). With tsr_model='none' no weights are set (hop count)."""
    if tsr_model == "none":
        for u, v in G.edges():
            G[u][v]["tsr"] = 1.0
            G[u][v].pop("w", None)
        return
    caps = nx.get_edge_attributes(G, "cap")
    mean_cap = float(np.mean(list(caps.values())))
    if tsr_model == "capacity":
        for u, v in G.edges():
            t = 1.0 - math.exp(-(G[u][v]["cap"] / 2.0) / tsr_scale)
            G[u][v]["tsr"] = min(1 - EPS, max(EPS, t))
    elif tsr_model == "load":
        kk = sample_k if (sample_k and sample_k < G.number_of_nodes()) else None
        ebc = nx.edge_betweenness_centrality(G, k=kk, normalized=False,
                                             seed=(rng.randrange(10 ** 9) if rng else None))
        mean_ebc = float(np.mean(list(ebc.values()))) or 1.0
        for u, v in G.edges():
            e = ebc.get((u, v), ebc.get((v, u), 0.0))
            rho = kappa * (e / mean_ebc) * (mean_cap / G[u][v]["cap"])
            G[u][v]["tsr"] = min(1 - EPS, max(EPS, math.exp(-rho)))
    else:
        raise ValueError(tsr_model)
    for u, v in G.edges():
        G[u][v]["w"] = -math.log(G[u][v]["tsr"])


# ----------------------------------------------------------------------------- centrality
def _minmax(d):
    vals = np.array(list(d.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi - lo < 1e-15:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def sampled_closeness(G, k, wattr, rng):
    """Pivot-sampled closeness (Eppstein-Wang): k single-source searches instead of N. Used for large N."""
    nodes = list(G.nodes())
    pivots = rng.sample(nodes, k)
    acc = {v: 0.0 for v in nodes}
    for p in pivots:
        dist = (nx.single_source_dijkstra_path_length(G, p, weight=wattr) if wattr
                else nx.single_source_shortest_path_length(G, p))
        for v, d in dist.items():
            acc[v] += d
    n = len(nodes)
    return {v: (n - 1) / max(acc[v] * n / k, 1e-12) for v in nodes}


def composite_centrality(G, alpha, weighted, bet_k=None, rng=None):
    """C(V) = alpha*ClosenessN + (1-alpha)*BetweennessN, both min-max normalised over current nodes.
    bet_k (optional) switches closeness AND betweenness to k-pivot estimates (needed for N >~ 500)."""
    wattr = "w" if weighted else None
    k = bet_k if (bet_k and bet_k < G.number_of_nodes()) else None
    if k:
        clo = sampled_closeness(G, k, wattr, rng or random.Random(0))
    else:
        clo = nx.closeness_centrality(G, distance=wattr)          # exact (N-1)/sum d
    bet = nx.betweenness_centrality(G, k=k, weight=wattr, normalized=True,
                                    seed=(rng.randrange(10 ** 9) if rng else None))
    clo, bet = _minmax(clo), _minmax(bet)
    return {v: alpha * clo[v] + (1 - alpha) * bet[v] for v in G}


def s_scores(G, alpha, lam, weighted, bet_k=None, rng=None):
    C = composite_centrality(G, alpha, weighted, bet_k, rng)
    dmin = min(d for _, d in G.degree())
    return {v: C[v] - lam * (G.degree(v) - dmin) for v in G}


# ----------------------------------------------------------------------------- target choice
def _argmax_random_tie(scores, rng, exclude=()):
    best = max(s for v, s in scores.items() if v not in exclude)
    ties = [v for v, s in scores.items() if v not in exclude and s >= best - 1e-12]
    return rng.choice(ties)


def choose_targets(G, method, m, alpha, lam, rng, bet_k, recompute_each, tsr_args):
    """Return the m distinct existing nodes a new node should open channels to."""
    nodes = list(G.nodes())
    m = min(m, len(nodes))
    if method == "random":
        return rng.sample(nodes, m)
    if method == "pa":
        chosen, pool = [], {v: G.degree(v) for v in nodes}
        for _ in range(m):
            tot = sum(pool.values())
            r, acc = rng.random() * tot, 0.0
            for v, w in pool.items():
                acc += w
                if acc >= r:
                    chosen.append(v)
                    pool.pop(v)
                    break
        return chosen
    if method == "min_degree":
        deg = {v: G.degree(v) + rng.random() * 1e-3 for v in nodes}      # random tie-break
        return sorted(nodes, key=lambda v: deg[v])[:m]

    # S(V)-family
    if method == "greedy_central":
        lam = 0.0
    weighted = (method == "sv_tsr")
    if weighted:
        refresh_edge_weights(G, **tsr_args, rng=rng)
    if not recompute_each:
        sc = s_scores(G, alpha, lam, weighted, bet_k, rng)
        chosen = []
        for _ in range(m):
            v = _argmax_random_tie(sc, rng, exclude=chosen)
            chosen.append(v)
        return chosen
    # sequential greedy: attach a provisional node, recompute, pick next
    chosen, tmp = [], "__new__"
    H = G.copy()
    H.add_node(tmp)
    cap0 = float(np.mean([d["cap"] for _, _, d in G.edges(data=True)]))
    for _ in range(m):
        sc = s_scores(H, alpha, lam, weighted, bet_k, rng)
        sc.pop(tmp, None)
        v = _argmax_random_tie(sc, rng, exclude=chosen)
        chosen.append(v)
        H.add_edge(tmp, v, cap=cap0, tsr=1.0, **({"w": 0.0 + 1e-3} if weighted else {}))
    return chosen


# ----------------------------------------------------------------------------- growth loop
def random_seed_graph(n0, rng):
    """Random connected seed graph (random tree + n0//2 extra edges); depends only on rng."""
    G = nx.Graph()
    G.add_nodes_from(range(n0))
    order = list(range(n0))
    rng.shuffle(order)
    for i in range(1, n0):
        G.add_edge(order[i], order[rng.randrange(i)])
    extra = n0 // 2
    while extra > 0:
        u, v = rng.sample(range(n0), 2)
        if not G.has_edge(u, v):
            G.add_edge(u, v)
            extra -= 1
    return G


def grow(method, n, m=2, alpha=0.5, lam=None, dk_allowed=None, seed=0,
         cap_dist="lognormal", mean_cap=1000.0, lnd_edgelist=None,
         tsr_model="load", tsr_scale=None, kappa=0.3, bet_k=None, recompute_each=False,
         tsr_sample_k=None, seed_size=None):
    """Grow a network to n nodes. Returns an undirected nx.Graph with edge attrs cap/(tsr/w)."""
    assert method in METHODS, method
    rng = random.Random(seed)
    if lam is None:
        lam = (1 - alpha) / float(dk_allowed) if dk_allowed else 0.0      # paper's tuning rule
    if method in ("sv_tsr",) and tsr_model == "none":
        raise ValueError("sv_tsr needs a tsr_model other than 'none'")
    tsr_scale = tsr_scale or mean_cap
    cap = make_capacity_sampler(cap_dist, mean_cap, rng, lnd_edgelist)
    tsr_args = dict(tsr_model=tsr_model, tsr_scale=tsr_scale, kappa=kappa, sample_k=tsr_sample_k)

    n0 = seed_size or max(5, m + 2)
    # the seed graph depends only on `seed` and is identical across methods
    srng = random.Random(seed * 7919 + 13)
    scap = make_capacity_sampler(cap_dist, mean_cap, srng, lnd_edgelist)
    G = random_seed_graph(n0, srng)
    for u, v in G.edges():
        G[u][v]["cap"] = scap()

    for new in range(n0, n):
        targets = choose_targets(G, method, m, alpha, lam, rng, bet_k, recompute_each, tsr_args)
        G.add_node(new)
        for t in targets:
            G.add_edge(new, t, cap=cap())

    # capital normalisation: same mean capacity per channel in every method
    scale = mean_cap / np.mean([d["cap"] for _, _, d in G.edges(data=True)])
    for u, v in G.edges():
        G[u][v]["cap"] *= scale
    G.graph.update(dict(method=method, n=n, m=m, alpha=alpha, lam=lam, seed=seed,
                        dk_allowed=dk_allowed, tsr_model=tsr_model, cap_dist=cap_dist,
                        mean_cap=mean_cap))
    return G


def graph_to_dict(G):
    return {"meta": dict(G.graph),
            "n": G.number_of_nodes(),
            "edges": [[int(u), int(v), float(d["cap"])] for u, v, d in G.edges(data=True)]}


def graph_from_dict(d):
    G = nx.Graph()
    G.add_nodes_from(range(d["n"]))
    for u, v, c in d["edges"]:
        G.add_edge(int(u), int(v), cap=float(c))
    G.graph.update(d.get("meta", {}))
    return G
