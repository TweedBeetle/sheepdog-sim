# October 2026 re-evaluation

The September 2026 run reported one training run on 32 held-out fields, without intervals. This
directory re-measures it without changing any simulation code, for a technical report (German).

What it adds:

- **200 fresh held-out fields** (seeds 1,000,000–1,000,199) for every arm, at 100 and 30 sheep,
  plus a second independent set (2,000,000–2,000,199) as a cross-check. Training fields are
  `seed*7919 + gen*131 + i`, at most 348,130 for every run here; the September sets were
  9000–9015 and 9100–9115.
- **The untrained population**: all 48 generation-zero networks, each on the first 25 fresh fields.
- **Training replication**: the whole pipeline of `../chain.sh` (A6 attempt at 30 sheep, the
  ladder, the no-curriculum control) rerun under ES seeds 21, 31 and 41, each evaluated on the
  same 200 fresh fields.
- **Reproduction checks**: the stored September weights reproduce the recorded per-field results
  for 5 of 6 arms exactly (`probe_union.py`; the sixth was a hand-picked untrained network that
  was never recorded), and a seed-11 retrain reproduces all seven September champions bit for bit
  (`out/retrain-seed11-vs-stored.json`).
- **Statistics** (`summarize.py` -> `out/results.json`): percentile bootstrap (10,000, reseeded
  per comparison) for means, medians and paired differences, Wilson intervals for rates, exact
  McNemar for paired completion, a two-way cluster bootstrap for the untrained population;
  `compare_september.py` -> `out/september_vs_fresh.json` tests the September fields against the
  fresh ones per arm.

Headline (100 sheep, 200 fresh fields): the curriculum dog pens 67.3% (95% CI 61.6–72.6) and
finishes all 100 on 46.5% of fields; the published heuristic 39.8% and 10.5%. Across four
training runs the curriculum dog ranges 63–80% and 38–63%, and beats the heuristic on both
measures in every run.

## Reproduce

From the repository root, Python 3.13 and numpy 2.5 (what ran: 3.13.5, 2.5.3):

```
uv run --no-project reeval_2026_10/reeval.py evaluate --n 100 --fields 200
uv run --no-project reeval_2026_10/reeval.py evaluate --n 30 --fields 200
uv run --no-project reeval_2026_10/reeval.py population --n 100 --fields 25
uv run --no-project reeval_2026_10/reeval.py retrain --seed 11      # ~85 min on an M1 Max
uv run --no-project reeval_2026_10/reeval.py check-stored
uv run --no-project reeval_2026_10/summarize.py
uv run --no-project reeval_2026_10/compare_september.py
```

`driver.sh` / `driver2.sh` are the order the runs were made in (driver.sh stalled at its load gate
after the seed-11 retrain; driver2.sh finished the rest). `GATE=` (empty) skips the load gate.
Same fields and weights give identical numbers; the same ES seed gives identical weights.

| path | what |
|---|---|
| `weights_2026_09/` | the September champions and generation-zero population, with SHA256SUMS |
| `out/fresh-fields/`, `out/fresh-fields-2/` | per-field rows for every arm |
| `out/retrain-seed*/` | each retrain's champions, snapshots and per-generation logs |
| `out/replicate-seed*/` | each retrain's three champions on the 200 fresh fields |
| `out/results.json` | every statistic the report quotes |
