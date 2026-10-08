# Sheepdog sim — a flock that reacts, one dog with senses, and evolution

The material half of the m37 episode **sheepdog**. A top-down field, a flock of sheep running
the published shepherding model, a pen, and one dog steered by a small neural network whose
weights are found by evolution. Nothing tells the dog how to herd; the only thing it is ever
given is a number at the end of each attempt saying how that attempt went, and the record of
what those numbers were and what each one produced is the episode.

```bash
uv run --no-project test/run.py                      # the controls; they fail, never skip
uv run --no-project probe.py                         # cost + what the published technique does
uv run --no-project run.py attempts                  # the fitness ladder, A1..A6
uv run --no-project run.py ladder --init <champ.npy> # 10 / 30 / 100 / skittish / split
uv run --no-project run.py compare --theta <champ>   # the dog against the shepherd's technique
uv run --no-project run.py ablate --theta <champ>    # what collapses when a rule is removed
uv run --no-project run.py replay --theta <champ>    # a recorded attempt for the picture
```

| file | role |
|---|---|
| `world.py` | the field, the flock, the clock, the dog's senses. Batched: one call steps B fields |
| `policy.py` | the dog's brain, 266 weights, batched over a generation |
| `heuristic.py` | the shepherd's technique (collect / drive) as the comparison arm. Trains nothing |
| `episode.py` | one batch of attempts, and the fitness functions in the order they were written |
| `evolve.py` | (mu, lambda) evolution strategy, population chunked across processes |
| `run.py` | the CLI: attempts, ladder, compare, ablate, replay |
| `test/run.py` | the controls |

**Why batched.** At B=1 an episode costs about 300 ms and almost all of it is per-call overhead
on tiny arrays. A whole generation of 120 costs 5.6 s at 30 sheep, so the ladder and the
ablations are affordable. The flock's noise is counter-based per field, so a field's trajectory
does not depend on which batch or which process it landed in — the control asserts it, and it is
what makes a generation's ranking one fair comparison.

`runs/` is gitignored; the episode copies what it uses into `production/sheepdog/data/`.

## October 2026 re-evaluation

`reeval_2026_10/` re-measures the September result without changing any simulation code: every
arm on 200 fresh held-out fields (and a second independent set of 200), the untrained population,
three more training runs under other seeds, and a seed-11 retrain that reproduces the September
champions bit for bit. Scripts, per-field results, the September weights and the statistics are
all in that directory; its README has the commands. At 100 sheep the curriculum dog pens 67.3%
(95% CI 61.6–72.6) against 39.8% for the published heuristic.
