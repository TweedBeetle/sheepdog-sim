#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy"]
# ///
"""The two SENTENCES b06 compares, run over the sixteen held-out fields.

b06 says a hundred and eighty generations of "did you finish" gets to about twelve and a half
seconds, and a hundred and eighty generations of "did you finish, and how much clock was left"
gets to just under ten. Those two means are already in `what-training-bought.json`. What was NOT
in `production/sheepdog/data/` is the PER-FIELD pair behind them, and b06's picture needs it: the
scene shows ONE of those sixteen fields, so the note register has to be able to say whether the
ordering it draws is typical or picked, and a number on screen with no file is not sayable.

    uv run --no-project sims/sheepdog/b06_arms.py

Writes runs/b06-arms/result.json and copies it to production/sheepdog/data/b06-arms-over-16.json.
It is its own check: the two per-field arrays it records must average to the two figures
`what-training-bought.json` already publishes, or it refuses to write. That is a stronger receipt
than the count on its own -- it says these are the same sixteen fields and the same two champions
the spoken numbers came from, not a fresh evaluation that happens to agree.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from episode import rollout  # noqa: E402
from world import FieldSpec  # noqa: E402

DATA = ROOT / "production/sheepdog/data"
ARMS = [
    ("A0-all-or-nothing", "the sentence that asks only whether you finished"),
    ("A6-speed", "the sentence that also counts how much of the clock was left"),
]


def main() -> None:
    seeds: list[int] = json.loads((DATA / "randoms-per-dog-30.json").read_text())["held_out_field_seeds"]
    bought = json.loads((DATA / "what-training-bought.json").read_text())
    spec = FieldSpec(n_sheep=30)

    per_arm: dict[str, dict] = {}
    for arm, gloss in ARMS:
        theta = np.load(HERE / "runs" / f"attempt-{arm}" / "champion.npy")
        s = rollout(spec, seeds, theta=theta).summary()
        per_arm[arm] = {"sentence": gloss, "all_in_rate": s["all_in_rate"], "all_in_s": s["all_in_s"]}

    # the check that makes this a receipt rather than a second opinion
    for arm, _ in ARMS:
        got = per_arm[arm]["all_in_s"]
        if any(t is None for t in got):
            raise SystemExit(f"{arm} failed to finish a field; b06 speaks these arms as always finishing")
        mean = sum(got) / len(got)
        published = bought[f"{arm}/final"]["mean_time_to_pen_s"]
        if abs(mean - published) > 0.01:
            raise SystemExit(
                f"{arm}: these sixteen fields average {mean:.3f} s, but what-training-bought.json "
                f"publishes {published}. Different fields or a different champion; refusing to write.",
            )

    a = per_arm["A0-all-or-nothing"]["all_in_s"]
    b = per_arm["A6-speed"]["all_in_s"]
    faster = [seeds[i] for i, (x, y) in enumerate(zip(a, b)) if y < x]
    out = {
        "note": "the two SENTENCES of b06, each arm's 180-generation champion, on the sixteen "
                "held-out fields the spoken means average over. The per-field pair behind "
                "what-training-bought.json's 12.48 and 9.89, so b06 can say whether the one field "
                "it draws is typical.",
        "command": "uv run --no-project sims/sheepdog/b06_arms.py",
        "field_seeds": seeds,
        "field_seeds_from": "randoms-per-dog-30.json held_out_field_seeds",
        "n_sheep": spec.n_sheep,
        "arms": per_arm,
        "clock_counting_faster_on": len(faster),
        "of_fields": len(seeds),
        "clock_counting_faster_seeds": faster,
        "median_gap_s": round(statistics.median(x - y for x, y in zip(a, b)), 4),
        "means_reproduce_what_training_bought": {
            "A0-all-or-nothing": round(sum(a) / len(a), 4),
            "A6-speed": round(sum(b) / len(b), 4),
        },
    }
    runs = HERE / "runs/b06-arms"
    runs.mkdir(parents=True, exist_ok=True)
    (runs / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    (DATA / "b06-arms-over-16.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"clock-counting faster on {out['clock_counting_faster_on']} of {out['of_fields']}, "
          f"median gap {out['median_gap_s']} s")


if __name__ == "__main__":
    main()
