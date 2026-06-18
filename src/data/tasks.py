"""Task templates -> verifiable (context, question, answer, answer_type) tuples.

answer_type drives how the reward function checks correctness:
  - "numeric"  : single number, compared with small tolerance
  - "name"     : a single player name, normalized string compare
  - "set"      : a set of names, order-insensitive
  - "decision" : a single token from a small vocab (e.g. TD / FG)

Each task fn takes a seeded RNG and returns a dict. Difficulty scales with how
many fields the model must combine.
"""

import random
from dataclasses import dataclass

from data.generators import (
    generate_box_score, render_box_score,
    generate_game_state, render_game_state,
)


@dataclass
class Sample:
    context: str
    question: str
    answer: str
    answer_type: str
    kind: str


# --- box-score tasks -------------------------------------------------------

def task_scrimmage_total(rng: random.Random) -> Sample:
    """Total yards from scrimmage (rush + rec) for one named player."""
    box = generate_box_score(rng)
    p = rng.choice(box["players"])
    total = p["rush_yds"] + p["rec_yds"]
    return Sample(
        context=render_box_score(box),
        question=f"How many total yards from scrimmage (rushing + receiving) did {p['name']} have?",
        answer=str(total), answer_type="numeric", kind="scrimmage_total",
    )


def task_team_points(rng: random.Random) -> Sample:
    """Reconstruct total points from the scoring line."""
    box = generate_box_score(rng)
    return Sample(
        context=render_box_score(box),
        question="How many total points did the team score?",
        answer=str(box["scoring"]["points"]), answer_type="numeric", kind="team_points",
    )


def task_most_scrimmage(rng: random.Random) -> Sample:
    """Which player had the most yards from scrimmage (argmax over the table)."""
    box = generate_box_score(rng)
    best = max(box["players"], key=lambda p: p["rush_yds"] + p["rec_yds"])
    # avoid ties making this ambiguous
    totals = sorted((p["rush_yds"] + p["rec_yds"] for p in box["players"]), reverse=True)
    if len(totals) > 1 and totals[0] == totals[1]:
        return task_team_points(rng)  # fall back rather than emit an ambiguous label
    return Sample(
        context=render_box_score(box),
        question="Which player had the most total yards from scrimmage?",
        answer=best["name"], answer_type="name", kind="most_scrimmage",
    )


def task_hundred_yard_receivers(rng: random.Random) -> Sample:
    """Set membership: all players with >= 100 receiving yards."""
    box = generate_box_score(rng)
    qualifiers = [p["name"] for p in box["players"] if p["rec_yds"] >= 100]
    answer = ", ".join(sorted(qualifiers)) if qualifiers else "none"
    return Sample(
        context=render_box_score(box),
        question="List every player with 100 or more receiving yards (comma-separated, or 'none').",
        answer=answer, answer_type="set", kind="hundred_yd_rec",
    )


def task_total_touchdowns(rng: random.Random) -> Sample:
    """Sum rushing + receiving TDs across all players."""
    box = generate_box_score(rng)
    tds = sum(p["rush_td"] + p["rec_td"] for p in box["players"])
    return Sample(
        context=render_box_score(box),
        question="How many total touchdowns (rushing + receiving) did these players score combined?",
        answer=str(tds), answer_type="numeric", kind="total_tds",
    )


# --- game-state task -------------------------------------------------------

def task_td_or_fg(rng: random.Random) -> Sample:
    """Decision: does the offense need a TD, or is a FG enough to take the lead?
    Trailing by 1-3 -> FG ties/leads (FG = 3, sufficient to lead only if down <=2;
    down by 3 a FG only ties). We define 'take the lead' strictly:
        need TD if deficit >= 3 (FG can't put you ahead), else FG suffices.
    """
    gs = generate_game_state(rng)
    deficit = abs(gs["score_diff"])
    answer = "FG" if deficit < 3 else "TD"
    return Sample(
        context=render_game_state(gs),
        question="To TAKE THE LEAD (not just tie), does the offense need a touchdown or is a field goal enough? Answer TD or FG.",
        answer=answer, answer_type="decision", kind="td_or_fg",
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
