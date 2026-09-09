"""Evolution: a population of dogs, and the best of them make the next generation.

A plain (mu, lambda) evolution strategy over the policy's weights: truncation selection,
gaussian mutation, one elite carried unchanged, and a step size that shrinks on a schedule. No
crossover, because the weights of a network are not independently meaningful and recombining
two of them mostly destroys both.

Every dog in a generation is judged on the SAME set of starting fields, so the ranking inside a
generation is a comparison and not a lottery, and the field seeds rotate between generations so
a dog cannot win by suiting one layout.

The population is simulated in chunks, one process per chunk, each chunk itself a batched
sim (world.py). The counter-based flock noise means a field's trajectory does not depend on
which chunk it landed in, so the parallel run and a single-process run give the same numbers;
test/run.py asserts that.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import policy
from episode import FITNESS, rollout
from world import HZ, FieldSpec

_G: dict = {}


def _init(spec_kw: dict, fitness: str):
    _G["spec"] = FieldSpec(**spec_kw)
    _G["fit"] = FITNESS[fitness]
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _chunk(args):
    """One flat batch of (dog, field) pairs. Flat rather than a loop over fields because the
    per-call overhead on small arrays is most of the cost, so the wider the single rollout the
    cheaper each episode in it."""
    theta, seeds = args
    r = rollout(_G["spec"], [int(s) for s in seeds], theta=np.asarray(theta))
    return _G["fit"](r)


@dataclass
class GenRow:
    gen: int
    best_fitness: float
    mean_fitness: float
    best_penned_frac: float
    best_all_in_rate: float
    best_spread: float
    sigma: float
    seconds: float


def train(spec: FieldSpec, fitness: str, gens: int, pop: int, seed: int,
          n_fields: int = 3, sigma0: float = 0.30, sigma_end: float = 0.06,
          elite_frac: float = 0.25, workers: int = 8, init: np.ndarray | None = None,
          out_dir: Path | None = None, keep_every: int = 20,
          quiet: bool = False) -> list[GenRow]:
    assert fitness in FITNESS, fitness
    rng = np.random.default_rng(seed)
    theta = policy.random_theta(rng, pop)
    if init is not None:
        theta[0] = init
        theta[1:] = init + rng.normal(0, sigma0, size=(pop - 1, policy.N_PARAMS))
    n_elite = max(2, int(pop * elite_frac))
    spec_kw = asdict(spec)
    log: list[GenRow] = []
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers, initializer=_init, initargs=(spec_kw, fitness)) as pool:
        for g in range(gens):
            t0 = time.time()
            sigma = sigma0 + (sigma_end - sigma0) * (g / max(1, gens - 1))
            fields = [int(seed * 7919 + g * 131 + i) for i in range(n_fields)]
            # every (dog, field) pair, flattened; dog index = row // n_fields
            th_flat = np.repeat(theta, n_fields, axis=0)
            sd_flat = np.tile(fields, pop)
            parts = np.array_split(np.arange(pop * n_fields), max(1, workers))
            flat = np.concatenate(pool.map(
                _chunk, [(th_flat[p], sd_flat[p]) for p in parts if len(p)], chunksize=1))
            scores = flat.reshape(pop, n_fields).mean(axis=1)
            order = np.argsort(-scores)
            best = theta[order[0]]
            # what the best dog DID, in the units the video speaks
            br = rollout(spec, fields, theta=best[None, :])
            log.append(GenRow(g, float(scores[order[0]]), float(scores.mean()),
                              float(br.fraction.mean()), float((br.all_in_step >= 0).mean()),
                              float(br.spread_mean.mean()), float(sigma),
                              round(time.time() - t0, 2)))
            if not quiet:
                r = log[-1]
                print(f"  g{g:04d} fit {r.best_fitness:9.3f} mean {r.mean_fitness:9.3f} "
                      f"penned {r.best_penned_frac:.2f} allin {r.best_all_in_rate:.2f} "
                      f"spread {r.best_spread:5.1f} sig {sigma:.3f} {r.seconds:5.1f}s",
                      flush=True)
            if out_dir and (g % keep_every == 0 or g == gens - 1):
                np.save(out_dir / f"g{g:04d}.npy", best)
            elites = theta[order[:n_elite]]
            kids = [best]
            while len(kids) < pop:
                kids.append(elites[rng.integers(n_elite)]
                            + rng.normal(0, sigma, size=policy.N_PARAMS))
            theta = np.stack(kids)
    if out_dir:
        (out_dir / "log.json").write_text(json.dumps([asdict(r) for r in log], indent=1) + "\n")
        np.save(out_dir / "champion.npy", theta[0])
    return log
