# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Statistics for the October 2026 re-evaluation: reads out/, writes out/results.json.

Intervals: 95 % percentile bootstrap over fields (10,000 resamples, fixed seed) for means and
medians and for paired differences, which are computed on the same fields; Wilson intervals for
rates; exact McNemar test (two-sided binomial on the discordant fields) for paired completion."""
from __future__ import annotations

import json, math, sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "out"
B = 10_000
RNG = np.random.default_rng(20261007)


def boot(x: np.ndarray, stat=np.mean) -> list[float]:
    # reseeded per call, so the same comparison gets the same interval wherever it appears
    idx = np.random.default_rng(20261007).integers(0, len(x), size=(B, len(x)))
    s = stat(x[idx], axis=1)
    return [round(float(np.percentile(s, 2.5)), 4), round(float(np.percentile(s, 97.5)), 4)]


def wilson(k: int, n: int) -> list[float]:
    if n == 0:
        return [float("nan")] * 2
    z, p = 1.959964, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def mcnemar(a: np.ndarray, b: np.ndarray) -> dict:
    n01 = int(((~a) & b).sum()); n10 = int((a & ~b).sum()); n = n01 + n10
    k = min(n01, n10)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"only_first": n10, "only_second": n01, "p_exact": float(f"{p:.3g}")}


def load(d: Path, n: int, arm: str):
    p = d / f"{n}-sheep__{arm}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())["rows"]
    rows.sort(key=lambda r: r["seed"])
    return {"seeds": np.array([r["seed"] for r in rows]),
            "frac": np.array([r["penned_frac"] for r in rows]),
            "allin": np.array([r["all_in"] for r in rows]),
            "gcm": np.array([r["gcm_arrived"] for r in rows]),
            "t": [r["all_in_s"] for r in rows]}


def describe(a) -> dict:
    n = len(a["frac"]); k = int(a["allin"].sum()); g = int(a["gcm"].sum())
    times = [t for t in a["t"] if t is not None]
    return {"fields": n,
            "penned_mean": round(float(a["frac"].mean()), 4), "penned_mean_ci": boot(a["frac"]),
            "penned_median": round(float(np.median(a["frac"])), 4),
            "penned_median_ci": boot(a["frac"], np.median),
            "all_in": k, "all_in_rate": round(k / n, 4), "all_in_ci": wilson(k, n),
            "zero_fields": int((a["frac"] == 0).sum()),
            "centre_reached": g, "centre_reached_rate": round(g / n, 4), "centre_reached_ci": wilson(g, n),
            "median_time_to_all_in_s": None if not times else round(float(np.median(times)), 2)}


def paired(a, b) -> dict:
    assert (a["seeds"] == b["seeds"]).all()
    d = a["frac"] - b["frac"]
    dc = a["allin"].astype(float) - b["allin"].astype(float)
    return {"penned_mean_diff": round(float(d.mean()), 4), "penned_mean_diff_ci": boot(d),
            "all_in_rate_diff": round(float(dc.mean()), 4), "all_in_rate_diff_ci": boot(dc),
            "mcnemar_all_in": mcnemar(a["allin"], b["allin"])}


def main():
    res: dict = {"note": __doc__.splitlines()[0], "bootstrap_resamples": B}
    ff = OUT / "fresh-fields"
    arms = ["laddered", "no-curriculum", "thirty-sheep", "heuristic", "heuristic-drive-only", "no-dog"]
    for n in (100, 30):
        got = {a: load(ff, n, a) for a in arms}
        got = {a: v for a, v in got.items() if v is not None}
        if not got:
            continue
        block = {"arms": {a: describe(v) for a, v in got.items()}, "paired": {}}
        for x, y in [("laddered", "heuristic"), ("no-curriculum", "heuristic"),
                     ("laddered", "no-curriculum"), ("heuristic", "heuristic-drive-only")]:
            if x in got and y in got:
                block["paired"][f"{x} - {y}"] = paired(got[x], got[y])
        res[f"fresh_fields_{n}_sheep"] = block
    ff2 = OUT / "fresh-fields-2"
    got2 = {a: load(ff2, 100, a) for a in arms}
    got2 = {a: v for a, v in got2.items() if v is not None}
    if got2:
        b2 = {"arms": {a: describe(v) for a, v in got2.items()}, "paired": {}}
        for x, y in [("laddered", "heuristic"), ("no-curriculum", "heuristic"),
                     ("laddered", "no-curriculum")]:
            if x in got2 and y in got2:
                b2["paired"][f"{x} - {y}"] = paired(got2[x], got2[y])
        res["second_fresh_set_100_sheep"] = b2
    pop = ff / "100-sheep__untrained-population.json"
    if pop.exists():
        P = json.loads(pop.read_text())
        per_dog = np.array([np.mean([r["penned_frac"] for r in rows]) for rows in P["per_dog"]])
        allin = np.array([np.mean([r["all_in"] for r in rows]) for rows in P["per_dog"]])
        M = np.array([[r["penned_frac"] for r in rows] for rows in P["per_dog"]])
        G = np.array([[r["gcm_arrived"] for r in rows] for rows in P["per_dog"]], dtype=float)
        # two-way cluster bootstrap: networks and fields resampled independently, because
        # attempts by one network (or on one field) are not independent of each other
        bs = [M[np.ix_(RNG.integers(0, M.shape[0], M.shape[0]),
                       RNG.integers(0, M.shape[1], M.shape[1]))].mean() for _ in range(B)]
        res["untrained_population_100_sheep"] = {
            "dogs": len(per_dog), "fields": P["fields"],
            "population_mean_penned": round(float(per_dog.mean()), 4),
            "population_mean_penned_ci": [round(float(np.percentile(bs, 2.5)), 4),
                                          round(float(np.percentile(bs, 97.5)), 4)],
            "attempts_all_in": int(sum(r["all_in"] for rows in P["per_dog"] for r in rows)),
            "attempts": int(M.size),
            "centre_reached_rate": round(float(G.mean()), 4),
            "per_dog_median": round(float(np.median(per_dog)), 4),
            "per_dog_max": round(float(per_dog.max()), 4),
            "dogs_above_half": int((per_dog > 0.5).sum()),
            "population_all_in_rate": round(float(allin.mean()), 4)}
    chk = OUT / "retrain-seed11-vs-stored.json"
    if chk.exists():
        res["retrain_seed11_vs_stored"] = json.loads(chk.read_text())
    reps = {}
    heur = load(ff, 100, "heuristic")
    stored = {a: load(ff, 100, a) for a in ("laddered", "no-curriculum", "thirty-sheep")}
    if all(v is not None for v in stored.values()):
        reps["11 (September run)"] = stored
    for d in sorted(OUT.glob("replicate-seed*")):
        got = {a: load(d, 100, a) for a in ("laddered", "no-curriculum", "thirty-sheep")}
        if all(v is not None for v in got.values()):
            reps[d.name.replace("replicate-seed", "")] = got
    if reps:
        rr = {}
        for s, got in reps.items():
            rr[s] = {a: describe(v) for a, v in got.items()}
            if heur is not None:
                rr[s]["paired"] = {f"{a} - heuristic": paired(v, heur) for a, v in got.items()}
            rr[s].setdefault("paired", {})["laddered - no-curriculum"] = paired(got["laddered"], got["no-curriculum"])
        res["training_replications_100_sheep"] = rr
        for a in ("laddered", "no-curriculum", "thirty-sheep"):
            m = [rr[s][a]["penned_mean"] for s in rr]; c = [rr[s][a]["all_in_rate"] for s in rr]
            res.setdefault("across_training_seeds", {})[a] = {
                "seeds": list(rr), "penned_mean": m, "all_in_rate": c,
                "penned_mean_range": [min(m), max(m)], "all_in_rate_range": [min(c), max(c)]}
    (OUT / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
