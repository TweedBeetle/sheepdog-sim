"""The shepherd's technique, as the published model states it and as a handler describes it.

Two moves, and the rule that switches between them:

  COLLECT  when some sheep has strayed further from the middle of the flock than the model's
           own cohesion radius: go round to the far side of THAT sheep and push it back in.
  DRIVE    when the flock is together: get behind it on the line running from the pen through
           the flock, and push the whole thing down that line.

A working handler's commands name the same two things: casting out around the flock to gather
it, then walking it on. This module is the comparison arm. Nothing is trained against it and
the dog never sees it.
"""
from __future__ import annotations

import math

import numpy as np

from world import R_A, Batch, _unit

COLLECT_STANDOFF = R_A


def target_point(bt: Batch, collect: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Where the shepherd's technique would stand, per field, and which move that is
    (True = collect)."""
    spec = bt.spec
    gcm = bt.gcm()
    d = np.linalg.norm(bt.sheep - gcm[:, None, :], axis=2)
    collecting = d.max(axis=1) > spec.f_n
    stray = bt.sheep[np.arange(bt.B), np.argmax(d, axis=1)]
    p_collect = stray + _unit(stray - gcm) * COLLECT_STANDOFF
    standoff = R_A * math.sqrt(spec.n_sheep)
    p_drive = gcm + _unit(gcm - spec.pen) * standoff
    if not collect:
        # THE ABLATION ARM: the published technique with one of its two moves removed. It only
        # ever drives, from behind the flock's centre, and never goes round to fetch a straggler.
        return p_drive, np.zeros(bt.B, dtype=bool)
    return np.where(collecting[:, None], p_collect, p_drive), collecting


STOP_RADIUS = 3 * R_A   # the paper's own: the real dog rarely came closer, because close
                        # approach splits the flock


def act(bt: Batch, stop_rule: bool = True,
        collect: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p, collecting = target_point(bt, collect=collect)
    r = p - bt.dog
    want = np.arctan2(r[:, 1], r[:, 0])
    err = (want - bt.heading + math.pi) % (2 * math.pi) - math.pi
    turn = np.clip(err / bt.spec.max_turn, -1.0, 1.0)
    # slow down when the heading is badly wrong, so it turns on the spot rather than arcs wide
    speed = np.where(np.abs(err) < 0.9, 1.0, 0.35)
    if stop_rule:
        # stop dead if it is closer to any sheep than the paper's 3 r_a
        nearest = np.linalg.norm(bt.sheep - bt.dog[:, None, :], axis=2).min(axis=1)
        speed = np.where(nearest < STOP_RADIUS, 0.0, speed)
    return turn, speed, collecting
