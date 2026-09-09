"""The field: flocks of sheep that react to each other and to one dog, a pen, and a clock.

The flock is the published shepherding model (Strombom et al. 2014, J. R. Soc. Interface
11:20140719 -- the citation is verified in production/sheepdog/data/citation.md before any of
it is spoken). Every sheep does the same four things each step: it runs from the dog when the
dog is close enough to see, it drifts toward the middle of its nearest neighbours, it pushes
off any neighbour that is too close, and it keeps a little of last step's direction. There is
no leader and no plan.

EVERYTHING IS BATCHED. One call steps B independent fields at once, because a generation of
dogs is B fields that differ only in the brain steering the dog: at B=1 an episode costs about
300 ms here and almost all of it is per-call overhead on tiny arrays, so a whole generation of
120 costs barely more than one episode. That is what makes the ladder and the ablations
affordable. B=1 is an ordinary case of the same code, so the replay a scene renders and the
episode a fitness score came from ran through one implementation.

Determinism: the batch carries one generator seeded per run; the same seeds and the same
weights give byte-identical trajectories. test/run.py asserts it, and asserts that a batch of
identical fields stays identical to a batch of one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# --- flock parameters ---------------------------------------------------------------------
# The published model's own values, kept as published so the flock is not tuned to flatter the
# dog. A change here is a change to the SUBJECT and is recorded, never silent.
R_A = 2.0          # a sheep pushes off a neighbour closer than this
RHO_A = 2.0        # how hard it pushes
C_ATTRACT = 1.05   # how hard it drifts toward its neighbours' centre
RHO_S = 1.0        # how hard it runs from the dog
H_INERTIA = 0.5    # how much of last step's direction it keeps
E_NOISE = 0.3      # angular noise
DELTA_SHEEP = 1.0  # a sheep's step length when it is moving
GRAZE_P = 0.05     # chance of a drifting step when the dog is out of sight
N_NEIGH = 50       # how many nearest neighbours a sheep tracks

HZ = 20            # sim steps per second of sim time; the video speaks seconds, not steps


@dataclass(frozen=True)
class FieldSpec:
    n_sheep: int = 30
    size: float = 150.0
    r_s: float = 65.0              # how close the dog must be before a sheep reacts to it
    dog_speed: float = 1.5         # the dog's step length at full speed
    max_turn: float = 0.6          # radians of heading change per step
    pen_x: float = 10.0
    pen_y: float = 10.0
    pen_r: float = 12.0            # a sheep inside this circle is penned
    steps: int = 1200              # the clock: 1200 steps at 20 Hz is one minute
    spawn_min: float = 60.0        # sheep spawn no closer to the pen than this
    skittish: float = 1.0          # multiplies r_s: >1 means the flock breaks earlier
    flee: float = 1.0              # multiplies rho_s ONLY. Setting it to 0 leaves a sheep able
                                   # to SEE the dog, so it still gathers toward its neighbours,
                                   # and removes only the push away from the dog. Setting r_s to
                                   # 0 instead removes both at once, because in this model the
                                   # gathering is conditioned on seeing the shepherd, and an
                                   # ablation that removes two things cannot say which one
                                   # mattered.
    split: bool = False            # spawn the flock as two separated groups

    @property
    def f_n(self) -> float:
        """The published model's own cohesion scale: the radius inside which a flock of this
        size counts as together. It is the thing the shepherd's technique switches on."""
        return R_A * self.n_sheep ** (2.0 / 3.0)

    @property
    def pen(self) -> np.ndarray:
        return np.array([self.pen_x, self.pen_y], dtype=float)


@dataclass
class Batch:
    sheep: np.ndarray        # (B, N, 2)
    sheep_dir: np.ndarray    # (B, N, 2)
    dog: np.ndarray          # (B, 2)
    heading: np.ndarray      # (B,)
    t: int
    seeds: np.ndarray            # (B,) the field seeds; the flock's noise is derived from them
    spec: FieldSpec
    done: np.ndarray = None      # (B,) a field whose attempt is over: nothing in it moves again

    @property
    def B(self) -> int:
        return self.sheep.shape[0]

    def gcm(self) -> np.ndarray:                      # (B, 2)
        return self.sheep.mean(axis=1)

    def penned_mask(self) -> np.ndarray:              # (B, N) bool
        return np.linalg.norm(self.sheep - self.spec.pen, axis=2) < self.spec.pen_r

    def spread(self) -> np.ndarray:                   # (B,)
        return np.linalg.norm(self.sheep - self.gcm()[:, None, :], axis=2).mean(axis=1)

    def mean_dist(self) -> np.ndarray:                # (B,)
        return np.linalg.norm(self.sheep - self.spec.pen, axis=2).mean(axis=1)


_M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix(x: np.ndarray) -> np.ndarray:
    x = (x + np.uint64(0x9E3779B97F4A7C15)) & _M64
    x = ((x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & _M64
    x = ((x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & _M64
    return x ^ (x >> np.uint64(31))


def _noise(field_seeds: np.ndarray, t: int, n: int, salt: int) -> np.ndarray:
    """Uniforms in [0,1), shape (B, N), determined by (field seed, step, sheep, salt) alone.

    A counter rather than a stream, so a field's trajectory is the same whether it is simulated
    on its own or inside a batch of a hundred and twenty. That is what lets a generation be
    split across processes and still be one fair comparison, and it is what lets the replay a
    scene renders be the same run the score came from.
    """
    with np.errstate(over="ignore"):     # wraparound is the point of a mixing function
        base = _splitmix(field_seeds.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
                         + np.uint64(t) * np.uint64(0xD1342543DE82EF95) + np.uint64(salt))
        h = _splitmix(base[:, None]
                      + np.arange(n, dtype=np.uint64)[None, :] * np.uint64(0x85EBCA6B))
    return ((h >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)).astype(np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > 1e-9)


def reset(spec: FieldSpec, seeds: list[int]) -> Batch:
    """One field per seed. Two fields with the same seed are the same starting layout, which
    is what makes a generation's ranking a comparison rather than a lottery."""
    B, N = len(seeds), spec.n_sheep
    pen = spec.pen
    sheep = np.zeros((B, N, 2), dtype=np.float32)
    for b, s in enumerate(seeds):
        r = np.random.default_rng(s)
        if spec.split:
            cs = []
            while len(cs) < 2:
                c = r.uniform(spec.size * 0.35, spec.size * 0.9, size=2)
                if np.linalg.norm(c - pen) > spec.spawn_min and all(
                        np.linalg.norm(c - o) > spec.f_n * 2.5 for o in cs):
                    cs.append(c)
            h = N // 2
            sheep[b] = np.concatenate([cs[0] + r.normal(0, spec.f_n * 0.35, size=(h, 2)),
                                       cs[1] + r.normal(0, spec.f_n * 0.35, size=(N - h, 2))])
        else:
            while True:
                c = r.uniform(spec.size * 0.35, spec.size * 0.85, size=2)
                if np.linalg.norm(c - pen) > spec.spawn_min:
                    break
            sheep[b] = c + r.normal(0, spec.f_n * 0.5, size=(N, 2))
    sheep = np.clip(sheep, 2.0, spec.size - 2.0)
    # The dog starts on the far side of the flock from the pen: where a handler would send it,
    # and the one starting condition every arm shares.
    away = _unit(sheep.mean(axis=1) - pen)
    dog = np.clip(sheep.mean(axis=1) + away * 30.0, 2.0, spec.size - 2.0)
    to_pen = pen - dog
    heading = np.arctan2(to_pen[:, 1], to_pen[:, 0])
    return Batch(sheep.astype(np.float32), np.zeros((B, N, 2), dtype=np.float32),
                 dog.astype(np.float32), heading.astype(np.float32), 0,
                 np.asarray(seeds, dtype=np.uint64), spec,
                 np.zeros(B, dtype=bool))


def step(bt: Batch, turn: np.ndarray, speed: np.ndarray) -> Batch:
    """One tick. turn in [-1, 1] scales max_turn; speed in [0, 1] scales dog_speed."""
    spec = bt.spec
    B, N = bt.B, spec.n_sheep
    dog_before, heading_before = bt.dog.copy(), bt.heading.copy()
    # The dog moves first, so a sheep reacts to where the dog is now.
    bt.heading = (bt.heading + np.clip(turn, -1, 1) * spec.max_turn) % (2 * math.pi)
    v = np.stack([np.cos(bt.heading), np.sin(bt.heading)], axis=1)
    bt.dog = np.clip(bt.dog + v * np.clip(speed, 0, 1)[:, None] * spec.dog_speed, 0.0, spec.size)

    s = bt.sheep
    to_dog = s - bt.dog[:, None, :]
    sees = np.linalg.norm(to_dog, axis=2) < spec.r_s * spec.skittish       # (B, N)

    d = s[:, :, None, :] - s[:, None, :, :]                               # (B, N, N, 2)
    dist2 = np.einsum("bijk,bijk->bij", d, d)                             # squared, no sqrt
    idx = np.arange(N)
    dist2[:, idx, idx] = np.inf

    k = min(N_NEIGH, N - 1)
    if k >= N - 1 and N > 1:
        # every other sheep is a neighbour, so the local centre of mass is the global one with
        # this sheep taken out: O(N) instead of a partition over the whole distance matrix
        lcm = (s.sum(axis=1, keepdims=True) - s) / (N - 1)
    elif k > 0:
        nb = np.argpartition(dist2, k - 1, axis=2)[:, :, :k]               # (B, N, k)
        lcm = np.take_along_axis(s[:, None, :, :], nb[..., None], axis=2).mean(axis=2)
    else:
        lcm = s.copy()

    seen = sees[..., None]
    force = np.where(seen, RHO_S * spec.flee * _unit(to_dog) + C_ATTRACT * _unit(lcm - s), 0.0)

    close = (dist2 < R_A * R_A) & (dist2 > np.float32(1e-9))
    if close.any():
        # unit vectors only for the pairs that are actually touching
        inv = np.zeros_like(dist2)
        np.divide(1.0, np.sqrt(dist2, where=close, out=np.ones_like(dist2)),
                  out=inv, where=close)
        rep = np.einsum("bijk,bij->bik", d, inv)
        force = force + RHO_A * _unit(rep)

    force = force + H_INERTIA * bt.sheep_dir
    ang = _noise(bt.seeds, bt.t, N, 1) * np.float32(2 * math.pi)
    force = force + E_NOISE * np.stack([np.cos(ang), np.sin(ang)], axis=2)

    moving = sees | (_noise(bt.seeds, bt.t, N, 2) < GRAZE_P)
    new_dir = np.where(moving[..., None], _unit(force), 0.0)
    new_sheep = np.clip(s + new_dir * DELTA_SHEEP, 0.0, spec.size)

    # An attempt that has succeeded is OVER, and a finished field freezes exactly as it stood.
    # Without this the clock keeps running on a solved field: the dog's target point collapses
    # onto the flock it has just penned, it walks into the middle of them, and the flock it
    # spent ten seconds gathering is scattered again by the time the minute is up. That is not
    # the shepherding task and it is not what any arm should be scored on -- the published
    # heuristic in particular has no hold behaviour and was never meant to, so scoring it on
    # holding would be measuring an absence its author never claimed to fill.
    d3 = bt.done[:, None, None]
    bt.sheep_dir = np.where(d3, bt.sheep_dir, new_dir)
    bt.sheep = np.where(d3, s, new_sheep)
    bt.dog = np.where(bt.done[:, None], dog_before, bt.dog)
    bt.heading = np.where(bt.done, heading_before, bt.heading)
    bt.t += 1
    return bt


# --- the dog's senses ------------------------------------------------------------------------
N_SECTORS = 8
OBS_DIM = N_SECTORS + 11


def observe(bt: Batch) -> np.ndarray:                                     # (B, OBS_DIM)
    """What the dog can tell about the world, in its own frame.

    The sector fan is the load-bearing design choice and the one the narration explains: the
    dog gets ONE number per direction, the nearness of the closest sheep there, so how much it
    senses does not change when the flock does. That is what lets a dog raised on thirty sheep
    be dropped in front of a hundred at all.
    """
    spec, B, N = bt.spec, bt.B, bt.spec.n_sheep
    rel = bt.sheep - bt.dog[:, None, :]
    dd = np.linalg.norm(rel, axis=2)
    ang = (np.arctan2(rel[:, :, 1], rel[:, :, 0]) - bt.heading[:, None]) % (2 * math.pi)
    sec = np.minimum((ang / (2 * math.pi) * N_SECTORS).astype(np.int64), N_SECTORS - 1)
    near = 1.0 / (1.0 + dd / spec.f_n)                                    # 1 underfoot, 0 far
    fan = np.zeros((B, N_SECTORS))
    np.maximum.at(fan, (np.repeat(np.arange(B), N), sec.ravel()), near.ravel())

    def bearing(target: np.ndarray):
        r = target - bt.dog
        n = np.linalg.norm(r, axis=1)
        a = np.arctan2(r[:, 1], r[:, 0]) - bt.heading
        return np.cos(a), np.sin(a), n / spec.size

    gcm = bt.gcm()
    d_gcm = np.linalg.norm(bt.sheep - gcm[:, None, :], axis=2)
    far = bt.sheep[np.arange(B), np.argmax(d_gcm, axis=1)]
    gc, gs, gd = bearing(gcm)
    pc, ps, pd = bearing(spec.pen)
    fc, fs, _ = bearing(far)
    spread = d_gcm.mean(axis=1) / spec.f_n
    penned = bt.penned_mask().mean(axis=1)
    left = np.full(B, 1.0 - bt.t / spec.steps)
    return np.stack([*fan.T, gc, gs, gd, pc, ps, pd, fc, fs, spread, penned, left], axis=1)
