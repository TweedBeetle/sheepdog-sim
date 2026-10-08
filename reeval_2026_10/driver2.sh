#!/usr/bin/env bash
# The rest of the October 2026 re-evaluation after driver.sh stalled at its gate (its `python3`
# resolved to a deleted uv build directory). Same steps, same order, gate on an absolute path.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="uv run --no-project reeval_2026_10/reeval.py"
W=${WORKERS:-8}
PY=${GATE_PYTHON:-python3}
GATE=${GATE-reeval_2026_10/machine_gate.py}   # GATE= (empty) skips the load gate
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
gate() { [ -z "$GATE" ] && return 0; while ! "$PY" "$GATE"; do say "machine busy, waiting"; sleep 60; done; }
step() { gate; say "START $*"; "$@"; say "END rc=$? $*"; }

step $R check-stored
for S in 21 31 41; do
  step $R retrain --seed $S --workers $W
  D=reeval_2026_10/out/retrain-seed$S
  step $R evaluate --n 100 --fields 200 --workers $W --name replicate-seed$S \
       --theta laddered=$D/ladder-r5-100-sheep/champion.npy \
               no-curriculum=$D/fresh-100/champion.npy \
               thirty-sheep=$D/attempt-A6/champion.npy
done
# a second, independent set of 200 fields for the stored arms, to tell whether the September
# fields or the first fresh set is the unusual one for the heuristic
step $R evaluate --n 100 --fields 200 --workers $W --start 2000000 --name fresh-fields-2
say "DRIVER2 DONE"
