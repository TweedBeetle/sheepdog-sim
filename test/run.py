# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""Controls for the sheepdog sim. Every one of these FAILS the suite, never skips.

They exist because three claims the video will make rest on them: that a run is reproducible,
that a generation's ranking is a fair comparison however the work was split across processes,
and that the measurements have the range to tell a good dog from a bad one at all (the project's
own ablation rule: if a score survives removing the thing it claims to measure, it measured
nothing).
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import heuristic, policy
from episode import FITNESS, rollout
from world import FieldSpec, observe, reset, step

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILS.append(name)


spec = FieldSpec(n_sheep=20, steps=400)
rng = np.random.default_rng(7)
th = policy.random_theta(rng, 4)

# 1. determinism: same seeds, same weights, same trajectory
a = rollout(spec, [1, 2, 3, 4], theta=th)
b = rollout(spec, [1, 2, 3, 4], theta=th)
check("determinism", np.array_equal(a.penned, b.penned) and
      np.allclose(a.dist_area, b.dist_area, atol=0, rtol=0))

# 2. batch independence: a field's outcome must not depend on who it was batched with.
#    This is what lets a generation be split across processes and stay one comparison.
solo = rollout(spec, [3], theta=th[2:3])
inb = rollout(spec, [1, 2, 3, 4], theta=th)
check("batch independence", abs(float(solo.dist_area[0] - inb.dist_area[2])) < 1e-4,
      f"solo {solo.dist_area[0]:.6f} vs in-batch {inb.dist_area[2]:.6f}")

# 3. the do-nothing control: with no dog, no sheep reaches the pen. If this ever pens a sheep
#    the pen is reachable by drift and every penned number below means less than it says.
still = rollout(FieldSpec(n_sheep=30), list(range(12)), still=True)
check("no dog pens nothing", float(still.penned.sum()) == 0.0,
      f"penned {still.penned.sum():.0f}")

# 4. dynamic range: the published shepherd's technique must beat a random dog by a distance.
#    An instrument that scores them the same cannot say anything about a trained one.
#    Two arms, because the paper's shepherd carries a rule ours does not: it stops dead within
#    3 r_a of any sheep. On the paper's OWN criterion (get the flock's centre to the target) the
#    published arm succeeds every time either way; on ours (every sheep through a gate) the stop
#    rule is what holds it back, and that difference is a result the episode reports.
sp = FieldSpec(n_sheep=30)
hh = rollout(sp, list(range(12)), use_heuristic=True, stop_rule=False)
hp = rollout(sp, list(range(12)), use_heuristic=True, stop_rule=True)
rr = rollout(sp, list(range(12)), theta=policy.random_theta(np.random.default_rng(1), 12))
check("dynamic range: shepherd >> random", hh.fraction.mean() - rr.fraction.mean() > 0.5,
      f"shepherd {hh.fraction.mean():.2f} vs random {rr.fraction.mean():.2f}")
check("published arm meets the PAPER's own criterion every time",
      float(hp.gcm_arrived.mean()) == 1.0 and float(hh.gcm_arrived.mean()) == 1.0,
      f"gcm arrived: with stop rule {hp.gcm_arrived.mean():.2f}, without {hh.gcm_arrived.mean():.2f}")
check("published arm solves thirty sheep outright", float(hp.fraction.mean()) == 1.0,
      f"penned {hp.fraction.mean():.2f}")
# and at a hundred it does not, which is the rung the episode is built on. The stop rule is a
# large part of why: it is the paper's own rule and it is what keeps the shepherd off the
# stragglers. Measured at 12 fields.
h100p = rollout(FieldSpec(n_sheep=100), list(range(12)), use_heuristic=True, stop_rule=True)
h100 = rollout(FieldSpec(n_sheep=100), list(range(12)), use_heuristic=True, stop_rule=False)
check("published arm does NOT solve a hundred sheep in the minute",
      float(h100p.fraction.mean()) < 0.95 and float((h100p.all_in_step >= 0).mean()) < 0.5,
      f"penned {h100p.fraction.mean():.2f}, all in on {(h100p.all_in_step>=0).mean():.2f} of fields")
check("the stop rule costs it at a hundred, where it binds",
      float(h100p.fraction.mean()) < float(h100.fraction.mean()) - 0.2,
      f"penned with stop rule {h100p.fraction.mean():.2f} vs without {h100.fraction.mean():.2f}")

# 5. every fitness function separates them in the same direction. A fitness that ranks a random
#    dog above the published technique is telling the search to find the wrong thing.
for name, f in FITNESS.items():
    check(f"fitness {name} ranks shepherd above random",
          float(f(hh).mean()) > float(f(rr).mean()),
          f"{float(f(hh).mean()):.2f} vs {float(f(rr).mean()):.2f}")

# 6. the property the whole subject rests on, and it is the published model's, not ours: a
#    sheep only gathers toward its neighbours while it can see the dog. So a dog in range
#    TIGHTENS the flock and a dog out of range leaves it to drift apart. Written the other way
#    round first, and the control failing is what caught it -- the video says this out loud, so
#    it is asserted here rather than assumed.
sp2 = FieldSpec(n_sheep=30, steps=400)
near = reset(sp2, [5]); far = reset(sp2, [5])
s0 = float(near.spread()[0])
near.dog = near.gcm().copy() + np.float32(20.0)
far.dog = np.array([[3.0, 148.0]], dtype=np.float32)
for _ in range(400):
    step(near, np.zeros(1), np.zeros(1)); step(far, np.zeros(1), np.zeros(1))
check("a dog in range gathers the flock; out of range it drifts apart",
      near.spread()[0] < s0 and far.spread()[0] > near.spread()[0] * 1.5,
      f"start {s0:.1f} -> in-range {near.spread()[0]:.1f}, out-of-range {far.spread()[0]:.1f}")

# 7. the senses are the size the design claims, and their size does not change with the flock.
o10 = observe(reset(FieldSpec(n_sheep=10), [1]))
o100 = observe(reset(FieldSpec(n_sheep=100), [1]))
check("sensor width is independent of flock size",
      o10.shape == o100.shape and o10.shape[1] == 19, f"{o10.shape} {o100.shape}")
check("senses are finite", bool(np.isfinite(o10).all() and np.isfinite(o100).all()))

# 8. the heuristic really does both moves, and switches on the published cohesion radius.
bt = reset(FieldSpec(n_sheep=30), [1])
bt.sheep[0, 0] = np.array([140.0, 140.0], dtype=np.float32)   # one sheep dragged far out
_, collecting_far = heuristic.target_point(bt)
bt2 = reset(FieldSpec(n_sheep=30), [1])
bt2.sheep[0] = bt2.gcm()[0] + np.zeros_like(bt2.sheep[0])     # every sheep on the centre
_, collecting_tight = heuristic.target_point(bt2)
check("shepherd collects when a sheep strays, drives when it does not",
      bool(collecting_far[0]) and not bool(collecting_tight[0]))

# 9. a mutation-style control: break the dog's steering and the shepherd arm must collapse.
class _Broken:
    pass
broken = rollout(sp, list(range(8)), theta=np.zeros((1, policy.N_PARAMS)))
check("a dead policy pens far less than the shepherd",
      broken.fraction.mean() < hh.fraction.mean() - 0.4,
      f"dead {broken.fraction.mean():.2f} vs shepherd {hh.fraction.mean():.2f}")

print()
if FAILS:
    print(f"{len(FAILS)} FAILED: {', '.join(FAILS)}")
    sys.exit(1)
print("all controls pass")
