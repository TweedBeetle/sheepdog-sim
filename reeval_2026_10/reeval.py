# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Re-evaluation of the sheepdog simulation, October 2026.

The September 2026 record reports one training run, measured on 32 held-out fields, with no
intervals. This script adds, without changing any simulation code:

  evaluate   every arm on a FRESH held-out field set (seeds that no run has used), with
             per-field rows written out so every statistic below can be recomputed
  retrain    the training pipeline from scratch under a new evolution-strategy seed:
             the 30-sheep attempt (A6), the ladder, and the no-curriculum control, with the
             exact settings of chain.sh. Seed 11 is the original run and must reproduce the
             stored weights bit for bit.
  summarize  bootstrap intervals, paired differences and the across-run spread

Outputs land in reeval_2026_10/out/. Nothing under runs/ is read except the stored weights.
"""
from __future__ import annotations

import argparse, json, math, multiprocessing as mp, os, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SIM = HERE.parent
sys.path.insert(0, str(SIM))
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

import policy
from episode import rollout
from evolve import train
from world import HZ, FieldSpec

OUT = HERE / "out"
FRESH0 = 1_000_000            # fresh held-out fields: 1,000,000 .. 1,000,000+n-1. Training
                              # fields are seed*7919 + gen*131 + i, below 400,000 for every
                              # training seed used here, and the earlier held-out sets are
                              # 9000-9115, so there is no overlap with anything trained or tuned.
W09 = HERE / "weights_2026_09"     # byte-identical copies of the September runs/ champions
STORED = {
    "laddered": W09 / "ladder-r5-100-sheep.npy",
    "no-curriculum": W09 / "fresh-100.npy",
    "thirty-sheep": W09 / "attempt-A6-speed.npy",
}


def _env() -> dict:
    return {"numpy": np.__version__, "python": platform.python_version(),
            "machine": platform.machine(), "date": time.strftime("%Y-%m-%d %H:%M %Z")}


def _rows(n_sheep: int, seeds: list[int], arm: dict) -> list[dict]:
    spec = FieldSpec(n_sheep=n_sheep)
    kw = {}
    if arm["kind"] == "theta":
        kw["theta"] = np.load(SIM / arm["path"])[None, :]
    elif arm["kind"] == "pop":
        kw["theta"] = np.load(SIM / arm["path"])[arm["index"]][None, :]
    elif arm["kind"] == "heuristic":
        kw.update(use_heuristic=True, collect=arm.get("collect", True))
    elif arm["kind"] == "still":
        kw["still"] = True
    r = rollout(spec, seeds, **kw)
    return [{"seed": int(s), "penned_frac": round(float(f), 4),
             "all_in": bool(a >= 0), "all_in_s": None if a < 0 else round(float(a) / HZ, 2),
             "gcm_arrived": bool(g)}
            for s, f, a, g in zip(seeds, r.fraction, r.all_in_step, r.gcm_arrived)]


def _job(args):
    n_sheep, seeds, arm = args
    return _rows(n_sheep, seeds, arm)


def run_arm(n_sheep: int, seeds: list[int], arm: dict, workers: int, chunk: int = 10) -> list[dict]:
    parts = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
    if workers <= 1:
        res = [_job((n_sheep, p, arm)) for p in parts]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(_job, [(n_sheep, p, arm) for p in parts], chunksize=1)
    return [row for part in res for row in part]


def cmd_evaluate(a):
    seeds = list(range(a.start, a.start + a.fields))
    arms: dict[str, dict] = {}
    if a.theta:                                   # a retrained run's champions
        for spec in a.theta:
            name, path = spec.split("=", 1)
            arms[name] = {"kind": "theta", "path": str(Path(path).resolve().relative_to(SIM))}
    else:
        arms = {name: {"kind": "theta", "path": str(p.relative_to(SIM))} for name, p in STORED.items()}
        arms["heuristic"] = {"kind": "heuristic", "collect": True}
        arms["heuristic-drive-only"] = {"kind": "heuristic", "collect": False}
        arms["no-dog"] = {"kind": "still"}
    out = OUT / a.name
    out.mkdir(parents=True, exist_ok=True)
    for name, arm in arms.items():
        p = out / f"{a.n}-sheep__{name}.json"
        if p.exists() and not a.force:
            print(f"skip {p.name} (exists)", flush=True); continue
        t0 = time.time()
        rows = run_arm(a.n, seeds, arm, a.workers)
        p.write_text(json.dumps({"arm": name, "n_sheep": a.n, "fields": len(seeds),
                                 "seeds": [seeds[0], seeds[-1]], "controller": arm,
                                 "env": _env(), "seconds": round(time.time() - t0, 1),
                                 "rows": rows}, indent=1) + "\n")
        m = np.mean([r["penned_frac"] for r in rows]); c = np.mean([r["all_in"] for r in rows])
        print(f"{a.n:3d} sheep  {name:22s} penned {m:.3f}  all-in {c:.3f}  "
              f"({time.time() - t0:.0f}s)", flush=True)


def cmd_population(a):
    """The untrained control: every network of the stored generation-zero population, so the
    floor is the population and not one network picked from it."""
    pop_path = W09 / "randoms-population.npy"
    pop = np.load(pop_path)
    seeds = list(range(FRESH0, FRESH0 + a.fields))
    out = OUT / a.name; out.mkdir(parents=True, exist_ok=True)
    p = out / f"{a.n}-sheep__untrained-population.json"
    t0 = time.time()
    per_dog = []
    for i in range(len(pop)):
        rows = run_arm(a.n, seeds, {"kind": "pop", "path": str(pop_path.relative_to(SIM)), "index": i}, a.workers)
        per_dog.append(rows)
        print(f"dog {i:2d}: penned {np.mean([r['penned_frac'] for r in rows]):.3f}", flush=True)
    p.write_text(json.dumps({"arm": "untrained-population", "n_sheep": a.n, "fields": len(seeds),
                             "seeds": [seeds[0], seeds[-1]], "population": len(pop),
                             "env": _env(), "seconds": round(time.time() - t0, 1),
                             "per_dog": per_dog}) + "\n")


def cmd_retrain(a):
    """chain.sh's training, verbatim in its settings, under ES seed a.seed."""
    d = OUT / f"retrain-seed{a.seed}"
    d.mkdir(parents=True, exist_ok=True)
    stages = {
        "attempt-A6": dict(spec=FieldSpec(n_sheep=30), gens=180, pop=48, n_fields=3, init=None),
    }
    t_all = time.time()
    steps = a.steps.split(",")
    if "attempt" in steps:
        t0 = time.time()
        train(FieldSpec(n_sheep=30), "A6-speed", gens=180, pop=48, seed=a.seed, n_fields=3,
              workers=a.workers, out_dir=d / "attempt-A6", quiet=True)
        print(f"attempt-A6 done ({time.time() - t0:.0f}s)", flush=True)
    if "ladder" in steps:
        cur = np.load(d / "attempt-A6" / "champion.npy")
        for name, kw in [("r1-10-sheep", dict(n_sheep=10)), ("r2-30-sheep", dict(n_sheep=30)),
                         ("r3-skittish", dict(n_sheep=30, skittish=1.6)),
                         ("r4-split", dict(n_sheep=30, split=True)), ("r5-100-sheep", dict(n_sheep=100))]:
            hard = name.endswith("100-sheep")
            t0 = time.time()
            train(FieldSpec(**kw), "A6-speed", gens=75 if hard else 25, pop=32 if hard else 48,
                  seed=a.seed, n_fields=2 if hard else 3, workers=a.workers, init=cur,
                  out_dir=d / f"ladder-{name}", quiet=True)
            cur = np.load(d / f"ladder-{name}" / "champion.npy")
            print(f"ladder {name} done ({time.time() - t0:.0f}s)", flush=True)
    if "fresh" in steps:
        t0 = time.time()
        train(FieldSpec(n_sheep=100), "A6-speed", gens=75, pop=32, seed=a.seed, n_fields=2,
              workers=a.workers, out_dir=d / "fresh-100", quiet=True)
        print(f"fresh-100 done ({time.time() - t0:.0f}s)", flush=True)
    (d / "env.json").write_text(json.dumps({**_env(), "seed": a.seed, "steps": steps,
                                            "seconds": round(time.time() - t_all, 1)}, indent=1) + "\n")


def cmd_check_stored(a):
    """Seed 11 is the September run: its retrained weights must equal the stored ones
    (weights_2026_09/, byte-identical copies of the September runs/ champions)."""
    d = OUT / "retrain-seed11"
    pairs = [("attempt-A6/champion.npy", "attempt-A6-speed.npy")]
    pairs += [(f"ladder-{r}/champion.npy", f"ladder-{r}.npy")
              for r in ["r1-10-sheep", "r2-30-sheep", "r3-skittish", "r4-split", "r5-100-sheep"]]
    pairs += [("fresh-100/champion.npy", "fresh-100.npy")]
    res = {}
    for mine, stored in pairs:
        pm, ps = d / mine, W09 / stored
        if not pm.exists():
            res[mine] = "not run"; continue
        x, y = np.load(pm), np.load(ps)
        res[mine] = {"identical": bool(np.array_equal(x, y)),
                     "max_abs_diff": float(np.max(np.abs(x - y)))}
    (OUT / "retrain-seed11-vs-stored.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("evaluate"); s.set_defaults(fn=cmd_evaluate)
    s.add_argument("--n", type=int, default=100); s.add_argument("--fields", type=int, default=200)
    s.add_argument("--name", default="fresh-fields"); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--theta", nargs="*"); s.add_argument("--force", action="store_true")
    s.add_argument("--start", type=int, default=FRESH0,
                   help="first field seed; a second independent set uses 2,000,000")
    s = sub.add_parser("population"); s.set_defaults(fn=cmd_population)
    s.add_argument("--n", type=int, default=100); s.add_argument("--fields", type=int, default=25)
    s.add_argument("--name", default="fresh-fields"); s.add_argument("--workers", type=int, default=4)
    s = sub.add_parser("retrain"); s.set_defaults(fn=cmd_retrain)
    s.add_argument("--seed", type=int, required=True); s.add_argument("--workers", type=int, default=4)
    s.add_argument("--steps", default="attempt,ladder,fresh")
    s = sub.add_parser("check-stored"); s.set_defaults(fn=cmd_check_stored)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
