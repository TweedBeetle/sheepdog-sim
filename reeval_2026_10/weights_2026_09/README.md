# The September 2026 weights

Copies of the trained networks the September run left in `runs/` (gitignored), so the October
re-evaluation can be reproduced from the repository alone. Each `.npy` is one champion's 266
weights (`policy.py`); `randoms-population.npy` is the 48 generation-zero networks.

| file | made by (chain.sh, seed 11) |
|---|---|
| attempt-A6-speed.npy | `run.py attempts --gens 180 --pop 48 --fields 3 --seed 11`, fitness A6, 30 sheep |
| attempt-A1-penned.npy | same command, fitness A1 |
| ladder-r1-10-sheep … ladder-r5-100-sheep.npy | `run.py ladder --init attempt-A6-speed --gens 25 --gens-hard 75 --pop 48 --pop-hard 32 --fields 3 --seed 11` |
| fresh-100.npy | `run.py fresh --rung 100 --gens 75 --pop 32 --fields 2 --seed 11` (no curriculum) |
| randoms-population.npy | `run.py randoms --pop 48 --seed 11` |

`SHA256SUMS` fixes the bytes. `reeval.py check-stored` retrains seed 11 and compares against
`runs/`, which these files copy.
