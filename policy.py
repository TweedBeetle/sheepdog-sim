"""The dog's brain: one small feedforward network, batched over a whole generation.

Two hidden layers would buy nothing here and cost search dimensions, so it is one. Keeping it
small is not an aesthetic: an evolution strategy's difficulty grows with the parameter count,
and every weight this policy does not have is search budget spent on the ones it does.
"""
from __future__ import annotations

import numpy as np

from world import OBS_DIM

HIDDEN = 12
SHAPES = [(OBS_DIM, HIDDEN), (HIDDEN,), (HIDDEN, 2), (2,)]
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES)


def unpack(theta: np.ndarray) -> list[np.ndarray]:
    """theta is (B, N_PARAMS) -> [w1 (B,I,H), b1 (B,H), w2 (B,H,2), b2 (B,2)]"""
    theta = np.atleast_2d(theta)
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(theta[:, i:i + n].reshape((theta.shape[0], *s)))
        i += n
    return out


def act(unpacked: list[np.ndarray], obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    w1, b1, w2, b2 = unpacked
    h = np.tanh(np.einsum("bi,bih->bh", obs, w1) + b1)
    o = np.tanh(np.einsum("bh,bho->bo", h, w2) + b2)
    return o[:, 0], (o[:, 1] + 1.0) * 0.5          # turn in [-1,1], speed in [0,1]


def random_theta(rng: np.random.Generator, n: int = 1, scale: float = 0.5) -> np.ndarray:
    return rng.normal(0.0, scale, size=(n, N_PARAMS))
