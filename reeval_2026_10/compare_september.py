# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Do the 32 September fields and the 200 fresh fields tell the same story, arm by arm?

Two-sided permutation test on the difference of mean penned fraction (20,000 permutations,
fixed seed). Writes out/september_vs_fresh.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
U = json.loads((HERE / "september_union_32_fields.json").read_text())["arms"]
PAIRS = [("shepherd", "heuristic"), ("laddered dog", "laddered"),
         ("no-curriculum dog", "no-curriculum"), ("thirty-sheep dog", "thirty-sheep"),
         ("shepherd, drive only", "heuristic-drive-only")]
rng = np.random.default_rng(20261007)
out = {}
for old, new in PAIRS:
    x = np.array(U[old]["per_field"])
    y = np.array([r["penned_frac"] for r in json.loads(
        (HERE / "out/fresh-fields" / f"100-sheep__{new}.json").read_text())["rows"]])
    obs = x.mean() - y.mean()
    allv = np.concatenate([x, y]); n = len(x); hits = 0
    for _ in range(20_000):
        p = rng.permutation(allv)
        hits += abs(p[:n].mean() - p[n:].mean()) >= abs(obs) - 1e-12
    out[new] = {"september_32_mean": round(float(x.mean()), 4), "fresh_200_mean": round(float(y.mean()), 4),
                "september_32_median": float(np.median(x)), "fresh_200_median": float(np.median(y)),
                "september_zero_share": round(float((x == 0).mean()), 4),
                "fresh_zero_share": round(float((y == 0).mean()), 4),
                "diff": round(float(obs), 4), "permutation_p": round(hits / 20_000, 4)}
(HERE / "out/september_vs_fresh.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
