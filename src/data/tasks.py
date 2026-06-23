"""Task templates -> verifiable (context, question, answer, answer_type) tuples.

answer_type drives how the reward function checks correctness:
  - "numeric"  : single number, compared with small tolerance
  - "name"     : a single player name, normalized string compare (last name ok)
  - "set"      : a set of names, order-insensitive
  - "decision" : a single token from a small vocab (e.g. TD / FG)

Each task fn takes a seeded RNG and returns a Sample. `depth` is the
compositional depth used by the Q3 taxonomy analysis (how many rows/ops the
answer requires); it travels with each row so eval can group by it.
"""

import random
from dataclasses import dataclass

from data.generators import (
    generate_box_score,
    generate_game_state,
    render_box_score,
    render_game_state,
)


@dataclass
class Sample:
    context: str
    question: str
    answer: str
    answer_type: str
    kind: str
    depth: int  # 1=low (single comparison) ... 4=high (filter/aggregate all rows)


# Compositional-depth labels (low -> high). Single source of truth shared by the
# dataset, the eval grouping, and EXPERIMENTS.md.
KIND_DEPTH = {
    "td_or_fg": 1,  # one comparison (deficit < 3); commit-early
    "scrimmage_total": 2,  # locate one player, add 2 fields
    "total_tds": 3,  # sum a field across all rows
    "most_scrimmage": 3,  # per-row total, then argmax
    "team_points": 4,  # sum player TDs (x6) + combine the FG/XP/2pt line
    "hundred_yd_rec": 4,  # filter all rows by threshold -> set
}


# --- box-score tasks -------------------------------------------------------


def task_scrimmage_total(rng: random.Random) -> Sample:
    """Total yards from scrimmage (rush + rec) for one named player."""
    box = generate_box_score(rng)
    p = rng.choice(box["players"])
    total = p["rush_yds"] + p["rec_yds"]
    return Sample(
        context=render_box_score(box),
        question=f"How many total yards from scrimmage (rushing + receiving) did {p['name']} have?",
        answer=str(total),
        answer_type="numeric",
        kind="scrimmage_total",
        depth=KIND_DEPTH["scrimmage_total"],
    )


def task_team_points(rng: random.Random) -> Sample:
    """Reconstruct total points: players' TDs (x6) + the FG/XP/2pt line.

    With the consistency invariant in generators.py the TD count is no longer
    printed, so this requires reading the player column AND the scoring line."""
    box = generate_box_score(rng)
    return Sample(
        context=render_box_score(box),
        question="How many total points did the team score?",
        answer=str(box["scoring"]["points"]),
        answer_type="numeric",
        kind="team_points",
        depth=KIND_DEPTH["team_points"],
    )


def task_most_scrimmage(rng: random.Random) -> Sample:
    """Which player had the most yards from scrimmage (argmax over the table).

    On a tie we RE-DRAW a fresh box rather than silently relabeling the sample
    as a different kind (which used to skew the realized kind distribution)."""
    box = generate_box_score(rng)
    for _ in range(64):  # bounded retries; ties are rare
        totals = sorted((p["rush_yds"] + p["rec_yds"] for p in box["players"]), reverse=True)
        if len(totals) < 2 or totals[0] != totals[1]:
            break
        box = generate_box_score(rng)
    best = max(box["players"], key=lambda p: (p["rush_yds"] + p["rec_yds"], p["name"]))
    return Sample(
        context=render_box_score(box),
        question="Which player had the most total yards from scrimmage?",
        answer=best["name"],
        answer_type="name",
        kind="most_scrimmage",
        depth=KIND_DEPTH["most_scrimmage"],
    )


def task_hundred_yard_receivers(rng: random.Random) -> Sample:
    """Set membership: all players with >= 100 receiving yards."""
    box = generate_box_score(rng)
    qualifiers = [p["name"] for p in box["players"] if p["rec_yds"] >= 100]
    answer = ", ".join(sorted(qualifiers)) if qualifiers else "none"
    return Sample(
        context=render_box_score(box),
        question="List every player with 100 or more receiving yards (comma-separated, or 'none').",
        answer=answer,
        answer_type="set",
        kind="hundred_yd_rec",
        depth=KIND_DEPTH["hundred_yd_rec"],
    )


def task_total_touchdowns(rng: random.Random) -> Sample:
    """Sum rushing + receiving TDs across all players."""
    box = generate_box_score(rng)
    tds = sum(p["rush_td"] + p["rec_td"] for p in box["players"])
    return Sample(
        context=render_box_score(box),
        question="How many total touchdowns (rushing + receiving) did these players score combined?",
        answer=str(tds),
        answer_type="numeric",
        kind="total_tds",
        depth=KIND_DEPTH["total_tds"],
    )


# --- game-state task -------------------------------------------------------


def task_td_or_fg(rng: random.Random) -> Sample:
    """Decision: does the offense need a TD, or is a FG enough to take the lead?
    A field goal is 3 points, so it takes the lead only if the deficit is 1 or 2;
    a deficit of 3 means a FG merely ties, so a TD is required.
        need TD if deficit >= 3, else FG suffices.
    The deficit is sampled ~50/50 between these cases (see generate_game_state)."""
    gs = generate_game_state(rng)
    deficit = abs(gs["score_diff"])
    answer = "FG" if deficit < 3 else "TD"
    return Sample(
        context=render_game_state(gs),
        question="To TAKE THE LEAD (not just tie), does the offense need a touchdown or is a field goal enough? Answer TD or FG.",
        answer=answer,
        answer_type="decision",
        kind="td_or_fg",
        depth=KIND_DEPTH["td_or_fg"],
    )


ALL_TASKS = [
    task_scrimmage_total,
    task_team_points,
    task_most_scrimmage,
    task_hundred_yard_receivers,
    task_total_touchdowns,
    task_td_or_fg,
]


def sample_one(rng: random.Random) -> Sample:
    return rng.choice(ALL_TASKS)(rng)
