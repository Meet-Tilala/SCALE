# Experiment kit: S(V) node-joining heuristic vs. current Lightning-Network growth

Validates the claims in *"Dynamic Network Balancing and Diameter Minimization"* (S(V) = C(V) − λ(Δ(V) − Δmin),
C = α·Closeness + (1−α)·Betweenness, λ = (1−α)/Δk_allowed, TSR-weighted distances w(e) = −log TSR_e).

Everything in the paper's Evaluation section runs with **Python only** (numpy, pandas, networkx, scipy, matplotlib, tabulate).
There are three experiments:

| Exp. | What | Scripts | Needs |
|---|---|---|---|
| **A** | Grow networks with S(V) and baselines; structure, load distribution, resilience | `run_structural.py`, `structural_metrics.py`, `sv_growth.py` | Python |
| **C** | Payment-level simulation on the grown N=50 topologies (**used in the paper**) | `run_pysim.py`, `payment_sim.py`, `make_workload.py` | Python |
| **S** | Sensitivity of the payment result to α and Δk (3×3 grid) | `run_sensitivity.py` | Python |
| B | Same payment study in the Spider/OMNeT++ simulator (**optional, not used in the paper**) | `run_spider.py`, `to_spider.py` | OMNeT++ 5.6 + Spider repo |

## Files
| File | Role |
|---|---|
| `sv_growth.py` | growth rules (`sv`, `sv_tsr`, `pa`, `random`, `greedy_central`, `min_degree`), TSR models, capacity sampling |
| `structural_metrics.py` | max degree, degree/betweenness Gini, top-10% share, diameter, Path TSR, LCC after attack, break-even ratio |
| `run_structural.py` | Exp. A driver, writes `results/structural.csv` (and JSON topologies with `--save-topologies`) |
| `payment_sim.py` | the custom payment simulator (see "Payment simulator" below) |
| `make_workload.py` | demand generator (circulation + optional DAG, log-normal sizes) shared by both payment simulators |
| `run_pysim.py` | Exp. C driver, reads `topologies/*.json`, writes `results/pysim.csv` (resumable) |
| `run_sensitivity.py` | Exp. S driver: grows topologies for a grid of (α, Δk), runs Exp. C on them, writes paired-difference tables |
| `analyze.py` | tables (md/tex/csv) and figures from `results/*.csv`; Wilcoxon tests vs. preferential attachment |
| `make_paper_tables.py` | booktabs tables used by `Evaluation.tex` (`paper_structural_n<N>.tex`, `paper_pysim_*.tex`) |
| `export_ccdf.py` | `data/ccdf_n300.csv` for the pgfplots degree-CCDF figure |
| `run_all.sh` | stage runner (Linux/macOS/WSL); see the note on `C-full` below |
| `test_sv.py` | 6 sanity tests; run before trusting any result |
| `run_spider.py`, `to_spider.py` | optional Spider/OMNeT++ route (Exp. B) |

## Hypotheses (each is a claim in the paper) and what would falsify it
| # | Claim | Metric | Falsified if |
|---|-------|--------|--------------|
| H1 | Degree is bounded / hubs do not form | `max_deg`, `deg_gini`, `frac_over_dk`, `gamma_hat` | max degree grows with N like preferential attachment |
| H2 | Diameter stays low | `diameter`, `avg_hops` vs N | diameter much larger than PA at equal edges |
| H3 | Routing revenue is spread more evenly | `bet_gini`, `bet_top10pct`, `bet_max_over_mean` | no reduction vs PA |
| H4 | Reliability: Ω_prop > Ω_current | payment success ratio (Exp. C) | S(V)-TSR not above PA in paired tests |
| H5 | TSR-awareness helps beyond hop-count S(V) | `sv_tsr` vs `sv` | no difference |
| Abl. | Both terms are needed | `greedy_central` (λ=0) and `min_degree` | one of them matches S(V) |
| Sens. | Conclusions do not hinge on α = 0.5, Δk = 15 | Exp. S tables; the (α, Δk) sweep in `structural.csv` | sign or significance of H1–H4 flips inside the grid |

Baselines: `pa` linear preferential attachment (= "current LN"), `random`, `greedy_central`, `min_degree`. All methods start from the
same random seed graph, give each joiner the same m channels and have the same mean capacity, so **edges and locked capital are
identical** (K_net,prop = K_net,current).

## Settings used in the paper
* Growth: seed graph n0 = 5, m = 2 channels per joiner, log-normal capacities (σ = 1), α = 0.5, Δk_allowed = 15 (λ = 1/30).
* Structural: N = 100–400, 20 seeds, centralities exact for N = 100 and pivot-sampled (`--bet-k 100`) for larger N.
* Payments (Exp. C): N = 50, 10 seeds, mean channel capacity c ∈ {5, 10, 20, 40} in payment-size units, schemes `sp1` and `ksp4`,
  400 s per run with the first 100 s discarded, 1 s in-flight hold, circulation demand (about 10 payments/s/node),
  payment sizes log-normal on [0.1, 10] (median 0.2, mean 0.85).
* Sensitivity (Exp. S): α ∈ {0.2, 0.5, 0.8} × Δk ∈ {8, 15, 30}; the structural part of this grid at N = 100 and 300 is already in `structural.csv`.

## Modelling choices to state in the paper
* **TSR is not observable at join time**, so it is modelled. Default `load`: TSR_e = exp(−κ·(edge-betweenness/mean)·(mean cap/cap_e)), κ = 0.3.
  `capacity`: TSR_e = 1 − exp(−cap_e/2 / scale). Every graph is *evaluated* under the same model, whichever rule grew it.
* Closeness/betweenness are min–max normalised over the current nodes at each step.
* **Payment simulator** (`payment_sim.py`): each channel's capacity is split 50/50 into two directional balances; a payment is tried on up to k
  hop-count shortest simple paths and uses the first with enough balance on every hop; funds are locked and credited to the reverse direction
  after the hold time; failed payments are dropped (no queueing, splitting, fees or rebalancing). Routing ignores TSR, so differences between
  growth rules come from topology only. The same demand is replayed on every topology of a given seed (paired).

## Setup
```
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python test_sv.py                                    # 6 "ok" lines
```
(Use `python3` instead of `python` on Linux/macOS.)

## Run order
```
# Exp. A: structure (N = 100, 300 with the 3x3 (alpha, dk) grid and 20 seeds, ~1-2 h)
python run_structural.py --preset paper
python run_structural.py --sizes 200 400 --seeds 20 --alphas 0.5 --dks 15 --bet-k 100 --append   # extra sizes, if needed

# Exp. C: payment simulation on the N=50 topologies (~10-20 min)
python run_structural.py --preset quick --sizes 50 --seeds 10 --alphas 0.5 --dks 15 \
    --save-topologies topologies --save-sizes 50 --out results/structural_n50.csv
python run_pysim.py --jobs 4 --mean-caps 5 10 20 40

# Exp. S: sensitivity to alpha and dk (3x3 grid, ~10-15 min); writes results_sens/ and topologies_sens/ only
python run_sensitivity.py --jobs 4

# tables and figures
python analyze.py structural --n 300 --alpha 0.5 --dk 15
python analyze.py spider --file pysim.csv --tag _py --n 50 --alpha 0.5 --dk 15
python make_paper_tables.py --n 300 --pysim-n 50 --alpha 0.5 --dk 15
python export_ccdf.py --n 300 --alpha 0.5 --dk 15
```
**`run_all.sh C-full` uses mean capacities 3, 10, 30, 100, which are not the paper's 5, 10, 20, 40.** Use the `run_pysim.py` line above for
paper numbers. Re-running `run_pysim.py` skips finished runs (cached in `results/pysim_runs/`); `results/pysim.csv` is rebuilt from that cache,
so keep the folder, or use a new `--results` directory.

## Exp. S output
`results_sens/sensitivity_<scheme>.csv` and `.tex`: for each (α, Δk) and each of S(V)-TSR and hop-count S(V), the mean paired success-ratio
difference vs. preferential attachment at every capacity, with `*` for Wilcoxon p < 0.05 (needs about 6 or more seeds to reach significance).
The (α = 0.5, Δk = 15) row should reproduce the main result (about +0.096 for S(V)-TSR, ksp4, c = 5). `structural_*.csv` hold the N = 50 structure per setting.

## Optional: Exp. B (Spider, OMNeT++)
`run_spider.py` simulates the same topologies in the Spider OMNeT++ simulator (`nitinawathare/LNBalanceRouting`), which has no node-joining
code, so topologies are grown here and exported with `to_spider.py`. It needs OMNeT++ 5.6 and a built `spiderNet` (the riskiest step: old C++
codebase). Stages `./run_all.sh B-smoke` and `./run_all.sh B-full`. Not used for the paper's numbers.

## Things a reviewer will ask
1. **Hard degree cap.** Δmin is the *current* minimum degree, so the penalty is relative; nothing forces max degree ≤ Δk_allowed. In the paper the
   plateau (24.6 for S(V)-TSR, 32.1 for hop-count S(V)) is above Δk = 15, and at α = 0.8 or Δk = 30 the maximum degree reaches or passes preferential attachment.
2. **Eq. (5) vs. low diameter.** Under the paper's revenue model Σ_v C_bet(v) = Σ_pairs (hops − 1), so shorter paths *lower* total routing revenue; the
   profitability argument rests on Ω, which is why the payment success ratio is the decisive measurement.
3. **Ω_prop > Ω_current** is tested, not proved; payment results are for N = 50 only and the structural reliability gain does not persist at N ≥ 200.
4. **Demand is uniform over nodes.** Real demand is skewed toward hubs/merchants; add a skewed-demand run (`--dag-pct`) as a robustness check.
5. **TSR proxies** and **pivot sampling** are approximations; state them. Joiners open m = 2 channels here, the protocol specifies one.
6. Fees, rebalancing and channel closures are not in the growth model or the simulator.
