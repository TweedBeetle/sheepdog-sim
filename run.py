# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""The sheepdog runs. Every result the episode may speak comes out of this file and lands in
runs/<name>/, and the episode copies what it uses into production/sheepdog/data/."""
from __future__ import annotations

import argparse, json, sys, time
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

import heuristic, policy
from episode import rollout
from evolve import train
from world import HZ, FieldSpec, reset

HERE = Path(__file__).parent
RUNS = HERE / "runs"

ATTEMPTS = ["A0-all-or-nothing", "A1-penned", "A2-closer", "A3-all-along", "A4-together",
            "A5-together-penned", "A6-speed"]


def _spec(**kw) -> FieldSpec:
    return FieldSpec(**kw)


def describe(spec: FieldSpec, seeds: list[int], theta=None, use_heuristic=False,
             still=False, collect=True, stop_rule=True) -> dict:
    r = rollout(spec, seeds, theta=theta, use_heuristic=use_heuristic, still=still,
                collect=collect, stop_rule=stop_rule)
    return r.summary()


# --- what the dog is DOING, measured against the shepherd's own two moves -------------------
def technique(spec: FieldSpec, seeds: list[int], theta: np.ndarray) -> dict:
    """Per step: which move the shepherd's technique would make here, and how far the dog is
    from where that technique would have it stand. Nothing here trains anything; it is the
    comparison the episode makes, and it is a measurement, not a judgement."""
    from world import observe, step
    bt = reset(spec, seeds)
    B = len(seeds)
    up = policy.unpack(theta)
    up = [np.repeat(x, B, axis=0) for x in up] if up[0].shape[0] == 1 else up
    d_c, d_d, n_c, n_d = 0.0, 0.0, 0, 0
    behind = 0
    for _ in range(spec.steps):
        p, collecting = heuristic.target_point(bt)
        gap = np.linalg.norm(bt.dog - p, axis=1) / spec.f_n
        d_c += float(gap[collecting].sum()); n_c += int(collecting.sum())
        d_d += float(gap[~collecting].sum()); n_d += int((~collecting).sum())
        # is the dog on the far side of the flock from the pen (the drive position), within 60 deg?
        gcm = bt.gcm()
        v1 = gcm - spec.pen; v2 = bt.dog - gcm
        cos = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
        behind += int((cos > 0.5).sum())
        turn, speed = policy.act(up, observe(bt))
        step(bt, turn, speed)
    tot = spec.steps * B
    return {"steps": tot,
            "collect_situations_frac": round(n_c / tot, 4),
            "gap_when_collect_f_n": round(d_c / max(1, n_c), 3),
            "gap_when_drive_f_n": round(d_d / max(1, n_d), 3),
            "behind_the_flock_frac": round(behind / tot, 4)}


def _save(name: str, obj) -> Path:
    d = RUNS / name; d.mkdir(parents=True, exist_ok=True)
    p = d / "result.json"; p.write_text(json.dumps(obj, indent=1) + "\n"); return p


def cmd_attempts(a):
    """The fitness ladder. Each attempt is a sentence telling the dog what the job is; the
    record of what each sentence produced is the episode's middle act."""
    import multiprocessing as mp
    which = a.only.split(",") if a.only else ATTEMPTS
    def one(fit):
        out = RUNS / f"attempt-{fit}"
        log = train(_spec(n_sheep=a.n), fit, gens=a.gens, pop=a.pop, seed=a.seed,
                    n_fields=a.fields, workers=a.workers, out_dir=out, quiet=True)
        champ = np.load(out / "champion.npy")[None, :]
        holdout = list(range(9_000, 9_012))
        res = {"fitness": fit, "gens": a.gens, "pop": a.pop, "n_fields": a.fields,
               "seed": a.seed, "n_sheep": a.n,
               "final": asdict(log[-1]), "best_of_log": asdict(max(log, key=lambda r: r.best_penned_frac)),
               "holdout": describe(_spec(n_sheep=a.n), holdout, theta=champ)}
        (out / "result.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"{fit}: penned {res['holdout']['penned_frac_mean']:.2f} on held-out fields "
              f"(all-in {res['holdout']['all_in_rate']:.2f})", flush=True)
    ctx = mp.get_context("fork")
    ps = [ctx.Process(target=one, args=(f,)) for f in which]
    for p in ps: p.start()
    for p in ps: p.join()


def cmd_ladder(a):
    init = np.load(a.init)
    rungs = [("r1-10-sheep", dict(n_sheep=10)),
             ("r2-30-sheep", dict(n_sheep=30)),
             ("r3-skittish", dict(n_sheep=30, skittish=1.6)),
             ("r4-split", dict(n_sheep=30, split=True)),
             ("r5-100-sheep", dict(n_sheep=100))]
    cur = init
    out_all = {}
    for name, kw in rungs:
        spec = _spec(**kw)
        before = describe(spec, list(range(9_000, 9_012)), theta=cur[None, :])
        d = RUNS / f"ladder-{name}"
        gens = a.gens_hard if (name.endswith("100-sheep") and a.gens_hard) else a.gens
        pop = a.pop_hard if (name.endswith("100-sheep") and a.pop_hard) else a.pop
        fields = 2 if name.endswith("100-sheep") else a.fields
        log = train(spec, a.fitness, gens=gens, pop=pop, seed=a.seed, n_fields=fields,
                    workers=a.workers, init=cur, out_dir=d, quiet=True)
        cur = np.load(d / "champion.npy")
        after = describe(spec, list(range(9_000, 9_012)), theta=cur[None, :])
        shep = describe(spec, list(range(9_000, 9_012)), use_heuristic=True)
        out_all[name] = {"spec": asdict(spec), "gens": gens, "pop": pop, "fields": fields,
                         "dog_before_this_rung": before, "dog_after": after, "shepherd": shep,
                         "technique_after": technique(spec, list(range(9_000, 9_006)), cur[None, :])}
        print(f"{name}: dog {before['penned_frac_mean']:.2f} -> {after['penned_frac_mean']:.2f}"
              f"   shepherd {shep['penned_frac_mean']:.2f}", flush=True)
        np.save(d / "champion.npy", cur)
    _save("ladder", out_all)


def cmd_compare(a):
    theta = np.load(a.theta)[None, :]
    seeds = list(range(9_000, 9_000 + a.seeds))
    out = {}
    for name, kw in [("10", dict(n_sheep=10)), ("30", dict(n_sheep=30)),
                     ("30-split", dict(n_sheep=30, split=True)), ("100", dict(n_sheep=100))]:
        spec = _spec(**kw)
        out[name] = {"dog": describe(spec, seeds, theta=theta),
                     "shepherd": describe(spec, seeds, use_heuristic=True),
                     "shepherd_drive_only": describe(spec, seeds, use_heuristic=True, collect=False),
                     "no_dog": describe(spec, seeds, still=True),
                     "dog_technique": technique(spec, seeds[:6], theta)}
        print(name, "dog", out[name]["dog"]["penned_frac_mean"],
              "shepherd", out[name]["shepherd"]["penned_frac_mean"], flush=True)
    _save("compare", out)


def cmd_ablate(a):
    """What collapses when one thing is removed. Each arm removes exactly ONE thing and changes
    nothing else, which is the only way a collapse says anything.

    The set is chosen by what the measurements made interesting rather than by what was planned:
    a hundred sheep turned out to defeat the published technique as thoroughly as it defeats us,
    so the informative rung is a SPLIT flock, where the published technique's second move is
    exactly the thing our dog has to find.
    """
    theta = np.load(a.theta)[None, :]
    seeds = list(range(9_100, 9_100 + a.seeds))
    split = _spec(n_sheep=30, split=True)
    base = _spec(n_sheep=30)
    out: dict = {}

    # 1. the two arms, on a split flock and on a single one
    out["dog_split"] = describe(split, seeds, theta=theta)
    out["dog_single"] = describe(base, seeds, theta=theta)
    out["shepherd_split"] = describe(split, seeds, use_heuristic=True)
    out["shepherd_single"] = describe(base, seeds, use_heuristic=True)

    # 2. THE PUBLISHED TECHNIQUE WITH ITS COLLECT MOVE REMOVED. Same code, same standoff, same
    #    stop rule; it only ever drives. If a split flock is what collecting is for, this is
    #    where the paper's own two-move structure earns its second move.
    out["shepherd_drive_only_split"] = describe(split, seeds, use_heuristic=True, collect=False)
    out["shepherd_drive_only_single"] = describe(base, seeds, use_heuristic=True, collect=False)

    # 3. THE FLEE RULE, REMOVED ON ITS OWN. Setting r_s to zero removes two things at once,
    #    because in this model a sheep only gathers toward its neighbours while it can SEE the
    #    dog: an ablation that removes both cannot say which one mattered, and the conclusion
    #    "everything works through the flee reaction" would be over-determined. `flee=0` leaves
    #    the sheep able to see the dog, and to gather, and removes only the push away from it.
    out["dog_no_flee"] = describe(_spec(n_sheep=30, flee=0.0), seeds, theta=theta)
    out["shepherd_no_flee"] = describe(_spec(n_sheep=30, flee=0.0), seeds, use_heuristic=True)
    # and the blunt version, kept beside it so the difference between the two is on the record
    out["dog_blind_flock"] = describe(_spec(n_sheep=30, r_s=0.0), seeds, theta=theta)
    out["shepherd_blind_flock"] = describe(_spec(n_sheep=30, r_s=0.0), seeds, use_heuristic=True)

    # 4. sheep that see the dog from anywhere, so there is no part of the field it is not
    #    disturbing and no way to work round the flock without moving it
    out["dog_always_flee"] = describe(_spec(n_sheep=30, r_s=400.0), seeds, theta=theta)

    # 5. the dog trained under the first countable sentence rather than the last one
    if a.naive:
        n = np.load(a.naive)[None, :]
        out["naive_sentence_dog_split"] = describe(split, seeds, theta=n)
        out["naive_sentence_dog_single"] = describe(base, seeds, theta=n)

    # 6. no curriculum: same sentence, same budget, straight in at the split rung
    if a.fresh:
        f = np.load(a.fresh)[None, :]
        out["no_curriculum_dog_split"] = describe(split, seeds, theta=f)

    # 7. the untrained control, so every number above has a floor under it
    out["untrained_median_split"] = describe(split, seeds, theta=np.zeros((1, policy.N_PARAMS)))
    out["no_dog_split"] = describe(split, seeds, still=True)

    for k, v in out.items():
        print(f"{k:32s} penned {v['penned_frac_mean']:.2f}  all-in {v['all_in_rate']:.2f}",
              flush=True)
    _save("ablate", out)


def cmd_fresh(a):
    """The no-curriculum control: same sentence, same budget as the whole ladder, straight in at
    the hardest rung with no easier ones before it. It is only a control if the budget matches,
    so its --gens is the ladder's rungs added together."""
    d = RUNS / f"fresh-{a.rung}"
    kw = {"split": dict(n_sheep=30, split=True), "100": dict(n_sheep=100)}[a.rung]
    train(_spec(**kw), a.fitness, gens=a.gens, pop=a.pop, seed=a.seed,
          n_fields=a.fields, workers=a.workers, out_dir=d, quiet=False)
    print("champion:", d / "champion.npy")


def cmd_randoms(a):
    """What an UNTRAINED population does, per rung, on held-out fields.

    This is the measurement that decides what the episode may claim. If a network made of random
    numbers already pens the flock, then "it learned to herd" is not a claim the runs support at
    that rung, and the piece has to say the true thing instead. Run before the script is written.
    """
    rng = np.random.default_rng(a.seed)
    theta = policy.random_theta(rng, a.pop)
    hold = list(range(9_100, 9_100 + a.seeds))
    out = {}
    rungs = [("10-sheep", dict(n_sheep=10)), ("30-sheep", dict(n_sheep=30)),
             ("100-sheep", dict(n_sheep=100)), ("skittish", dict(n_sheep=30, skittish=1.6)),
             ("split", dict(n_sheep=30, split=True))]
    for name, kw in rungs:
        spec = _spec(**kw)
        fr = np.zeros(a.pop)
        allin = np.zeros(a.pop)
        for s in hold:
            r = rollout(spec, [s] * a.pop, theta=theta)
            fr += r.fraction
            allin += (r.all_in_step >= 0)
        fr /= len(hold); allin /= len(hold)
        shep = describe(spec, hold, use_heuristic=True)
        out[name] = {
            "spec": asdict(spec), "population": a.pop, "held_out_fields": len(hold),
            "penned_frac": {"min": round(float(fr.min()), 4), "median": round(float(np.median(fr)), 4),
                            "mean": round(float(fr.mean()), 4), "max": round(float(fr.max()), 4)},
            "dogs_above_half": int((fr > 0.5).sum()),
            "dogs_above_ninety": int((fr > 0.9).sum()),
            "best_dog_index": int(np.argmax(fr)),
            "best_dog_all_in_rate": round(float(allin[int(np.argmax(fr))]), 4),
            "shepherd": shep,
        }
        print(f"{name:10s} untrained: median {np.median(fr):.2f} max {fr.max():.2f}  "
              f"{int((fr>0.9).sum())} of {a.pop} above 0.90   |  shepherd "
              f"{shep['penned_frac_mean']:.2f}", flush=True)
    d = RUNS / "randoms"; d.mkdir(parents=True, exist_ok=True)
    np.save(d / "population.npy", theta)
    _save("randoms", out)


def cmd_collect_sweep(a):
    """Where the published technique's SECOND move earns its place.

    Same code, same standoff, same stop rule, one thing removed: it only ever drives and never
    goes round to fetch a straggler. Run across flock sizes because the paper reports the share
    of time spent collecting rising with group size, so size is the axis the move lives on.
    """
    hold = list(range(9_100, 9_100 + a.seeds))
    rows = []
    for name, kw in (("10", dict(n_sheep=10)), ("30", dict(n_sheep=30)),
                     ("30-split", dict(n_sheep=30, split=True)), ("60", dict(n_sheep=60)),
                     ("100", dict(n_sheep=100)), ("150", dict(n_sheep=150))):
        spec = _spec(**kw)
        both = describe(spec, hold, use_heuristic=True, collect=True)
        drive = describe(spec, hold, use_heuristic=True, collect=False)
        rows.append({"condition": name, "spec": asdict(spec), "fields": len(hold),
                     "both_moves": both, "drive_only": drive,
                     "collect_share_of_steps": both["collect_frac"],
                     "cost_of_removing_collect": round(both["penned_frac_mean"] - drive["penned_frac_mean"], 4)})
        print(f"{name:10s} both {both['penned_frac_mean']:.2f} (collect {both['collect_frac']:.2f})"
              f"   drive only {drive['penned_frac_mean']:.2f}", flush=True)
    _save("collect-sweep", {"note": "the published technique with its collect move removed, "
                                    "across flock sizes; nothing else changed",
                            "rows": rows})


def cmd_gallery(a):
    """A whole generation on one field, recorded together: every dog in the population facing
    the same starting layout, which is what makes a generation's ranking a comparison.

    It also settles a thing the picture could otherwise imply wrongly. A population of fifty
    random dogs is not fifty useless dogs: most do nothing, and a few by luck do something. The
    gallery shows the distribution rather than the champion, so 'the first attempt' on screen is
    a typical one and not the best of fifty.
    """
    rng = np.random.default_rng(a.seed)                 # the same stream evolve.train uses
    theta = policy.random_theta(rng, a.pop)
    if a.theta:                                          # a later generation's saved population
        theta = np.load(a.theta)
    spec = _spec(n_sheep=a.n)
    r = rollout(spec, [a.field] * len(theta), theta=theta, record=True, record_every=a.every)
    order = np.argsort(-r.fraction)
    d = RUNS / "replays"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{a.name}.json"
    p.write_text(json.dumps({
        "name": a.name, "spec": asdict(spec), "seed": a.field, "record_every": a.every, "hz": HZ,
        "population": int(len(theta)),
        "penned_frac": [round(float(x), 4) for x in r.fraction],
        "rank": [int(i) for i in order],
        "summary": r.summary(), "frames": r.replay}, separators=(",", ":")))
    fr = np.sort(r.fraction)
    print(f"{p}  {len(r.replay)} frames  {len(theta)} dogs on field {a.field}")
    print(f"  penned fraction: min {fr[0]:.2f} median {np.median(fr):.2f} max {fr[-1]:.2f}; "
          f"{int((fr > 0.5).sum())} of {len(fr)} pen more than half")


def cmd_replay(a):
    spec = _spec(n_sheep=a.n, skittish=a.skittish, split=a.split, flee=a.flee)
    kw = {}
    if a.theta:
        th = np.load(a.theta)
        if th.ndim == 2:
            # a saved POPULATION rather than one dog: pick the one asked for, and default to the
            # median so that "a typical untrained dog" on screen is typical and not the best of
            # forty-eight, which would be a picture that lies
            hold = list(range(9_100, 9_108))
            fr = np.zeros(len(th))
            for s in hold:
                fr += rollout(_spec(n_sheep=a.n), [s] * len(th), theta=th).fraction
            order = np.argsort(fr)
            idx = int(order[a.rank]) if a.rank >= 0 else int(order[len(order) // 2])
            print(f"  population of {len(th)}: picked index {idx}, penned {fr[idx]/len(hold):.2f} "
                  f"(median {np.median(fr)/len(hold):.2f})")
            th = th[idx]
        kw["theta"] = th[None, :]
    elif a.heuristic:
        kw["use_heuristic"] = True
        kw["collect"] = not a.no_collect
    else:
        kw["still"] = True
    r = rollout(spec, [a.seed], record=True, record_every=a.every, **kw)
    d = RUNS / "replays"; d.mkdir(parents=True, exist_ok=True)
    p = d / f"{a.name}.json"
    p.write_text(json.dumps({"name": a.name, "spec": asdict(spec), "seed": a.seed,
                             "record_every": a.every, "hz": HZ,
                             "summary": r.summary(), "frames": r.replay}, separators=(",", ":")))
    print(f"{p}  {len(r.replay)} frames  penned {r.fraction[0]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("attempts", cmd_attempts), ("ladder", cmd_ladder), ("compare", cmd_compare),
                     ("ablate", cmd_ablate), ("replay", cmd_replay), ("fresh", cmd_fresh),
                     ("gallery", cmd_gallery), ("randoms", cmd_randoms),
                     ("collect-sweep", cmd_collect_sweep)]:
        s = sub.add_parser(name); s.set_defaults(fn=fn)
        s.add_argument("--seed", type=int, default=11)
        if name in ("attempts", "ladder", "fresh"):
            s.add_argument("--gens", type=int, default=180)
            s.add_argument("--pop", type=int, default=48)
            s.add_argument("--fields", type=int, default=3)
            s.add_argument("--workers", type=int, default=1)
        if name == "attempts":
            s.add_argument("--n", type=int, default=30); s.add_argument("--only")
        if name in ("ladder", "fresh"):
            s.add_argument("--fitness", default="A6-speed")
        if name == "fresh":
            s.add_argument("--rung", default="split", choices=["split", "100"])
        if name == "ladder":
            s.add_argument("--init", required=True)
            s.add_argument("--gens-hard", type=int, default=0)
            s.add_argument("--pop-hard", type=int, default=0)
        if name in ("compare", "ablate", "replay", "gallery"):
            s.add_argument("--theta")
        if name == "gallery":
            s.add_argument("--name", required=True); s.add_argument("--n", type=int, default=30)
            s.add_argument("--pop", type=int, default=48); s.add_argument("--every", type=int, default=6)
            s.add_argument("--field", type=int, default=9001)
        if name in ("compare", "ablate", "randoms", "collect-sweep"):
            s.add_argument("--seeds", type=int, default=12)
        if name == "randoms":
            s.add_argument("--pop", type=int, default=48)
        if name == "ablate":
            s.add_argument("--naive"); s.add_argument("--fresh")
        if name == "replay":
            s.add_argument("--name", required=True); s.add_argument("--n", type=int, default=30)
            s.add_argument("--every", type=int, default=2)
            s.add_argument("--heuristic", action="store_true")
            s.add_argument("--skittish", type=float, default=1.0)
            s.add_argument("--split", action="store_true")
            s.add_argument("--no-collect", action="store_true")
            s.add_argument("--flee", type=float, default=1.0)
            s.add_argument("--rank", type=int, default=-1)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
