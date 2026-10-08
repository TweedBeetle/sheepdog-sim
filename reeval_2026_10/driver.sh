#!/usr/bin/env bash
# The October 2026 re-evaluation, every step in order, one at a time. Log: out/driver.log.
# Between steps it waits until the machine passes the responsiveness gate it was given
# (ps median < 2 s, swap-ins < 2,000 per 15 s); a step never starts on a busy machine.
set -uo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="uv run --no-project reeval_2026_10/reeval.py"
W=${WORKERS:-4}
GATE=${GATE-reeval_2026_10/machine_gate.py}   # GATE= (empty) skips the load gate
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
gate() { [ -z "$GATE" ] && return 0; while ! python3 "$GATE"; do say "machine busy, waiting"; sleep 60; done; }
step() { gate; say "START $*"; "$@"; say "END rc=$? $*"; }

step $R evaluate --n 100 --fields 200 --workers $W
step $R evaluate --n 30  --fields 200 --workers $W
step $R population --n 100 --fields 25 --workers $W
step $R retrain --seed 11 --workers $W
step $R check-stored
for S in 21 31 41; do
  step $R retrain --seed $S --workers $W
  D=reeval_2026_10/out/retrain-seed$S
  step $R evaluate --n 100 --fields 200 --workers $W --name replicate-seed$S \
       --theta laddered=$D/ladder-r5-100-sheep/champion.npy \
               no-curriculum=$D/fresh-100/champion.npy \
               thirty-sheep=$D/attempt-A6/champion.npy
done
say "DRIVER DONE"
