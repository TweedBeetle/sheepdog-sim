# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Feasibility probe: does the world work, what does a generation cost, and what does the
published shepherd's technique achieve on it? Not content: the read that says the subject is
buildable, and the dynamic-range check that says the measurements can tell anything apart."""
from __future__ import annotations

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import policy
from episode import rollout
from world import FieldSpec

rng = np.random.default_rng(0)
seeds = list(range(8))
for n in (10, 30, 100):
    spec = FieldSpec(n_sheep=n)
    t0 = time.time(); h = rollout(spec, seeds, use_heuristic=True); th = time.time() - t0
    t0 = time.time(); s = rollout(spec, seeds, still=True); ts = time.time() - t0
    t0 = time.time(); r = rollout(spec, seeds, theta=policy.random_theta(rng, len(seeds))); tr = time.time() - t0
    print(f"N={n:3d}  shepherd penned {h.fraction.mean():.2f} all-in {h.summary()['all_in_rate']:.2f} "
          f"collect {h.collect_frac.mean():.2f} | random dog {r.fraction.mean():.2f} | "
          f"no dog {s.fraction.mean():.2f}   [{len(seeds)} fields: {th:.1f}/{tr:.1f}/{ts:.1f} s]")

# cost of one generation-shaped batch
for n, B in ((30, 120), (100, 120)):
    spec = FieldSpec(n_sheep=n)
    t0 = time.time()
    rollout(spec, list(range(B)), theta=policy.random_theta(rng, B))
    print(f"generation-shaped batch N={n} B={B}: {time.time()-t0:.1f} s")
print("policy params:", policy.N_PARAMS)
