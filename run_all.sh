#!/usr/bin/env bash
# Stage runner. Usage:  ./run_all.sh <stage>
#   check        run unit tests
#   A-quick      structural pilot (~2 min)
#   A-paper      structural sweep, N=100,300, 20 seeds (~1-2 h)
#   A-scale      N=1000, 5 seeds, appended to the same CSV (~1-2 h)
#   topos        write the N=50 topologies Spider will simulate (10 seeds, alpha=0.5, dk=15)
#   C-smoke      NO-OMNeT++ alternative: tiny pure-Python payment simulation (seconds)
#   C-full       NO-OMNeT++ alternative: full payment simulation on the N=50 topologies (~10-20 min)
#   B-smoke      tiny end-to-end Spider run (needs the built simulator)
#   B-full       full Spider experiment (hours; resumable)
#   analyze      tables + figures from results/*.csv
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
case "${1:-}" in
  check)    $PY test_sv.py ;;
  A-quick)  $PY run_structural.py --preset quick ;;
  A-paper)  $PY run_structural.py --preset paper ;;
  A-scale)  $PY run_structural.py --preset scale --append ;;
  topos)    $PY run_structural.py --preset quick --sizes 50 --seeds 10 --alphas 0.5 --dks 15 \
              --save-topologies topologies --save-sizes 50 --out results/structural_n50.csv ;;
  C-smoke)  $PY run_pysim.py --smoke --seeds 0 ;;
  C-full)   $PY run_pysim.py --jobs "${JOBS:-4}" --mean-caps 3 10 30 100
            $PY analyze.py spider --file pysim.csv --tag _py --n 50 --alpha 0.5 --dk 15 ;;
  B-smoke)  $PY run_spider.py --topo-dir topologies --smoke --methods pa sv_tsr --seeds 0 --jobs 1 ;;
  B-full)   $PY run_spider.py --topo-dir topologies --alpha 0.5 --dk 15 --jobs "${JOBS:-2}" \
              --schemes waterfilling shortestPath --mean-caps 500 2000 ;;
  analyze)  $PY analyze.py structural --n 100 --alpha 0.5 --dk 15
            $PY analyze.py structural --n 300 --alpha 0.5 --dk 15 || true
            $PY analyze.py spider --n 50 --alpha 0.5 --dk 15 || true ;;
  *) sed -n 2,12p "$0"; exit 1 ;;
esac
