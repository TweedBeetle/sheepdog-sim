# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Which stored weights produced which arm of production/sheepdog/data/hundred-sheep-union.json.

The union file was written in-session on 2026-09-07 without a committed script, so the mapping
from arm name to weights file is recovered here by re-running candidates and matching the
per-field arrays exactly."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
import numpy as np
from episode import rollout
from world import FieldSpec

# a copy of m37's production/sheepdog/data/hundred-sheep-union.json, so this runs from the repo alone
UNION = json.loads((HERE / "september_union_32_fields.json").read_text())
SEEDS = UNION["seeds"]
spec = FieldSpec(n_sheep=100)

def per_field(**kw):
    r = rollout(spec, SEEDS, **kw)
    return [round(float(x), 4) for x in r.fraction]

cands = {name: SIM / "runs" / name / "champion.npy" for name in
         ["ladder-r5-100-sheep", "fresh-100", "ladder-r4-split", "ladder-r2-30-sheep",
          "ladder-r3-skittish", "ladder-r1-10-sheep", "attempt-A6-speed", "attempt-A1-penned"]}
want = {k: v["per_field"] for k, v in UNION["arms"].items()}
found = {}
t0 = time.time()
for name, p in cands.items():
    got = per_field(theta=np.load(p)[None, :])
    hits = [arm for arm, pf in want.items() if pf == got]
    print(f"{name:22s} mean {np.mean(got):.4f}  matches {hits}  ({time.time()-t0:.0f}s)", flush=True)
    for h in hits: found[h] = f"runs/{name}/champion.npy"
for label, kw in [("heuristic", dict(use_heuristic=True)),
                  ("heuristic drive-only", dict(use_heuristic=True, collect=False)),
                  ("zeros", dict(theta=np.zeros((1, 266)))), ("no dog", dict(still=True))]:
    got = per_field(**kw)
    hits = [arm for arm, pf in want.items() if pf == got]
    print(f"{label:22s} mean {np.mean(got):.4f}  matches {hits}", flush=True)
    for h in hits: found[h] = label
pop = np.load(SIM / "runs/randoms/population.npy")
if "untrained dog" not in found:
    for i in range(len(pop)):
        got = per_field(theta=pop[i][None, :])
        if got == want["untrained dog"]:
            print(f"untrained dog = runs/randoms/population.npy[{i}]"); found["untrained dog"] = f"runs/randoms/population.npy[{i}]"; break
print(json.dumps(found, indent=1))
