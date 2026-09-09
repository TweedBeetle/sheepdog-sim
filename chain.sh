#!/usr/bin/env bash
# Everything after the fitness attempts, in order, unattended. Log: runs/chain.log
#
# The shape is what the measurements made it, not what was planned. A hundred sheep defeats the
# published technique as thoroughly as it defeats us, and the technique's collect move does
# nothing at all below sixty, so the ladder ends at the hundred rather than passing through it,
# and the no-curriculum control sits there too.
set -uo pipefail
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
R="uv run --no-project run.py"
say() { echo "[$(date +%H:%M:%S)] $*"; }

say "waiting for the attempts"
while pgrep -f "run.py attempts" >/dev/null; do sleep 20; done
say "attempts done"

CH=runs/attempt-A6-speed/champion.npy
[ -f "$CH" ] || { say "no A6 champion; stopping"; exit 1; }

# The collect sweep first: it is cheap, it needs nothing trained, and it is the measurement the
# whole third act rests on.
say "collect sweep"
$R collect-sweep --seeds 12

say "ladder from $CH"
$R ladder --init "$CH" --fitness A6-speed --gens 25 --gens-hard 75 --pop 48 --pop-hard 32 \
   --fields 3 --workers 8 --seed 11
say "ladder done"

# The no-curriculum control: the SAME sentence and the SAME number of generations at a hundred
# sheep, with no easier flocks before it. Not an equal-evaluations control -- the laddered arm
# also had a hundred generations at ten, thirty, skittish and split -- and the report says so.
say "fresh at a hundred, no curriculum"
$R fresh --rung 100 --fitness A6-speed --gens 75 --pop 32 --fields 2 --workers 8 --seed 11

L100=runs/ladder-r5-100-sheep/champion.npy
L30=runs/ladder-r2-30-sheep/champion.npy
say "compare"
$R compare --theta "$L100" --seeds 16
say "ablate"
$R ablate --theta "$L100" --seeds 16 --naive runs/attempt-A1-penned/champion.npy \
   --fresh runs/fresh-100/champion.npy
say "galleries"
$R gallery --name gen0-gallery-30  --n 30  --pop 48 --seed 11 --field 9001 --every 6
$R gallery --name gen0-gallery-100 --n 100 --pop 48 --seed 11 --field 9002 --every 8
say "replays"
$R replay --name untrained-typical --n 30  --theta runs/randoms/population.npy --seed 9001 --every 3
$R replay --name trained-30        --n 30  --theta "$CH" --seed 9001 --every 3
$R replay --name naive-a1-30       --n 30  --theta runs/attempt-A1-penned/champion.npy --seed 9001 --every 3
$R replay --name shepherd-30       --n 30  --heuristic --seed 9001 --every 3
$R replay --name shepherd-100      --n 100 --heuristic --seed 9002 --every 3
$R replay --name trained-100       --n 100 --theta "$L100" --seed 9002 --every 3
$R replay --name fresh-100         --n 100 --theta runs/fresh-100/champion.npy --seed 9002 --every 3
$R replay --name split-30          --n 30  --theta "$L30" --seed 9004 --every 3 --split
$R replay --name skittish-30       --n 30  --theta runs/ladder-r3-skittish/champion.npy --seed 9003 --every 3 --skittish 1.6
$R replay --name nodog-30          --n 30  --seed 9001 --every 4
say "CHAIN DONE"
