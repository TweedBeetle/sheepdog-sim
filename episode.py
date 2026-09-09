"""A batch of attempts: dogs, flocks, a clock, and what came of them.

Also the fitness functions. They are a NAMED, ORDERED list rather than one function with
knobs, because the order they were written in IS the record of what was tried and what it did.
production/sheepdog/data/attempts.md quotes this file.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import heuristic
import policy
from world import HZ, Batch, FieldSpec, observe, reset, step


@dataclass
class Results:
    """One row per field in the batch. Every column is a thing the picture can also show."""
    penned: np.ndarray            # (B,) sheep inside the pen at the final step
    n: int
    steps: int
    all_in_step: np.ndarray       # (B,) first step every sheep was inside, or -1
    mean_dist_end: np.ndarray     # (B,)
    dist_area: np.ndarray         # (B,) mean over time of the flock's mean distance to the pen
    spread_mean: np.ndarray       # (B,)
    collect_frac: np.ndarray      # (B,) heuristic arm only: share of steps spent collecting
    gcm_arrived: np.ndarray       # (B,) the PAPER's own criterion: flock centre reached the pen
    replay: list | None = None

    @property
    def fraction(self) -> np.ndarray:
        return self.penned / self.n

    def summary(self) -> dict:
        return {"n": self.n, "episodes": int(len(self.penned)),
                "penned_frac_mean": round(float(self.fraction.mean()), 4),
                "penned_frac_per_seed": [round(float(x), 4) for x in self.fraction],
                "all_in_s": [None if s < 0 else round(float(s) / HZ, 2) for s in self.all_in_step],
                "all_in_rate": round(float((self.all_in_step >= 0).mean()), 4),
                "mean_dist_end": round(float(self.mean_dist_end.mean()), 2),
                "spread_mean": round(float(self.spread_mean.mean()), 2),
                "gcm_arrived_rate": round(float(self.gcm_arrived.mean()), 4),
                "collect_frac": round(float(self.collect_frac.mean()), 4)}


def rollout(spec: FieldSpec, seeds: list[int], theta: np.ndarray | None = None,
            use_heuristic: bool = False, still: bool = False, stop_rule: bool = True,
            collect: bool = True, record: bool = False, record_every: int = 2) -> Results:
    """Run len(seeds) fields at once.

    Exactly one controller: `theta` (B, P) network weights, `use_heuristic`, or `still` (the
    do-nothing control, which is the ablation that proves the pen is not reachable by drift).
    """
    bt = reset(spec, seeds)
    B = bt.B
    up = policy.unpack(theta) if theta is not None else None
    if up is not None and up[0].shape[0] == 1 and B > 1:
        up = [np.repeat(x, B, axis=0) for x in up]

    dist_acc = np.zeros(B)
    spread_acc = np.zeros(B)
    collect_acc = np.zeros(B)
    all_in = np.full(B, -1)
    gcm_arrived = np.zeros(B, dtype=bool)
    frames = [] if record else None

    for t in range(spec.steps):
        if use_heuristic:
            turn, speed, collecting = heuristic.act(bt, stop_rule=stop_rule, collect=collect)
            collecting = collecting & ~bt.done
            collect_acc += collecting
        elif still:
            turn = np.zeros(B); speed = np.zeros(B); collecting = np.zeros(B, bool)
        else:
            turn, speed = policy.act(up, observe(bt))
            collecting = np.zeros(B, bool)
        if record and t % record_every == 0:
            # everything the picture needs comes from the sim, computed here rather than
            # recomputed in the renderer: a second implementation of the sensor fan or of the
            # shepherd's target point is a second thing that can disagree with the run
            fan = observe(bt)[:, :8]
            hp, hcol = heuristic.target_point(bt)
            frames.append({
                "t": t,
                "sheep": np.round(bt.sheep, 2).tolist(),
                "dog": np.round(bt.dog, 2).tolist(),
                "heading": np.round(bt.heading, 3).tolist(),
                "fan": np.round(fan, 3).tolist(),
                "collecting": collecting.astype(int).tolist(),
                "penned": bt.penned_mask().astype(int).tolist(),
                "shepherd_would": np.round(hp, 2).tolist(),
                "shepherd_move": hcol.astype(int).tolist(),
                "spread": np.round(bt.spread(), 2).tolist(),
                "gcm": np.round(bt.gcm(), 2).tolist(),
                "done": bt.done.astype(int).tolist(),
            })
        step(bt, turn, speed)
        dist_acc += bt.mean_dist()
        spread_acc += bt.spread()
        inside = bt.penned_mask().all(axis=1)
        all_in = np.where((all_in < 0) & inside, t, all_in)
        gcm_arrived |= np.linalg.norm(bt.gcm() - spec.pen, axis=1) < spec.pen_r
        # the attempt is over the moment every sheep is inside; the field freezes there, so
        # every number below is the state at success and a dog that finishes early keeps it
        bt.done |= inside
        # a recorded attempt stops a beat after it is over: the frozen tail is not a picture
        if record and bool(bt.done.all()) and t - int(all_in.max()) > 3 * HZ:
            break

    res = Results(penned=bt.penned_mask().sum(axis=1).astype(float), n=spec.n_sheep,
                  steps=spec.steps, all_in_step=all_in,
                  mean_dist_end=bt.mean_dist(), dist_area=dist_acc / spec.steps,
                  spread_mean=spread_acc / spec.steps, collect_frac=collect_acc / spec.steps,
                  gcm_arrived=gcm_arrived.astype(float))
    if record:
        res.replay = frames
    return res


# --- the fitness functions, in the order they were written ---------------------------------
# Each is an attempt at telling the dog what the job is. Larger is better in all of them.
# A dog is only ever as good as the sentence it was given, and these are the sentences.

def a0_all_or_nothing(r: Results) -> np.ndarray:
    """The first sentence anyone writes: did you do the job? One if every sheep is in the pen,
    zero otherwise. It is the task, stated exactly, with nothing added."""
    return (r.all_in_step >= 0).astype(float)

def a1_penned(r: Results) -> np.ndarray:
    """How many sheep ended up in the pen. Still only the goal, but counted rather than
    answered yes or no."""
    return r.penned

def a2_closer(r: Results) -> np.ndarray:
    """How close the flock ended up. A slope where A1 had a cliff."""
    return -r.mean_dist_end

def a3_all_along(r: Results) -> np.ndarray:
    """A2 averaged over the whole attempt, so getting there and holding it beats arriving in
    the last second."""
    return -r.dist_area

def a4_together(r: Results) -> np.ndarray:
    """A3 plus a term for keeping the flock together, which is what the collect move is for and
    what A3 has no reason to care about."""
    return -r.dist_area - 0.5 * r.spread_mean

def a5_together_penned(r: Results) -> np.ndarray:
    """A4 with the actual goal added back on top, weighted large enough to matter once the dog
    is close enough to ever score it."""
    return -r.dist_area - 0.5 * r.spread_mean + 40.0 * r.fraction

def a6_speed(r: Results) -> np.ndarray:
    """A5 plus a bonus for finishing early, so 'penned at all' stops being the ceiling."""
    bonus = np.where(r.all_in_step >= 0, 30.0 * (1.0 - r.all_in_step / r.steps), 0.0)
    return a5_together_penned(r) + bonus


FITNESS = {"A0-all-or-nothing": a0_all_or_nothing, "A1-penned": a1_penned, "A2-closer": a2_closer, "A3-all-along": a3_all_along,
           "A4-together": a4_together, "A5-together-penned": a5_together_penned,
           "A6-speed": a6_speed}
