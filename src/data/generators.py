"""Synthetic NFL data generators.

Why synthetic: it gives infinite, perfectly-labeled training data with zero
licensing/scraping issues, and lets you dial difficulty. The ground truth is
computed, not annotated, so the verifiable reward is exact by construction.

Swap in real data (nfl_data_py / nflverse, or NGS exports) for the EVAL set
later to make the headline number credible -- keep training synthetic.

Everything is seeded for reproducibility.

Design invariants (added in the study-hardening pass; see REVIEW.md):
  * Boxes are PHYSICALLY CONSISTENT: team points are derived from the players'
    own touchdowns plus the field-goal / extra-point / two-point line, so a model
    that reasons about football is never penalized for a contradiction. (Before,
    team TDs were an independent draw and disagreed with the player rows ~86% of
    the time.)
  * LAST NAMES ARE UNIQUE within a box, so "which player ..." answers are
    well-posed and a last-name-only answer is unambiguous. (Before, ~40% of boxes
    had a duplicated last name.)
  * The TD count is NOT printed as a separate aggregate, so `total_tds` and
    `team_points` still require reading the player column (no answer leak), while
    staying internally consistent.
"""

import random

FIRST_INITIALS = list("ABCDEFGHJKLMRST")
LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Wilson",
    "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee",
    "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark", "Ramirez",
    "Lewis", "Robinson", "Walker",
]


def _name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_INITIALS)}. {rng.choice(LAST_NAMES)}"


def _unique_names(rng: random.Random, n: int) -> list[str]:
    """n names with UNIQUE LAST NAMES, so "which player" answers are well-posed
    and a last-name-only response is unambiguous (the eval checker accepts it)."""
    names: list[str] = []
    used_last: set[str] = set()
    while len(names) < n:
        nm = _name(rng)
        last = nm.split()[-1]
        if last in used_last:
            continue
        used_last.add(last)
        names.append(nm)
    return names


def generate_box_score(rng: random.Random, n_skill: int = 6) -> dict:
    """A simplified offensive box score for a set of skill players plus team
    scoring. Player stat lines are independent draws; team scoring is DERIVED
    from the players' touchdowns so the box is internally consistent."""
    names = _unique_names(rng, n_skill)
    players = []
    for nm in names:
        rush_att = rng.choice([0, 0, 0, 3, 6, 10, 14, 18, 22])
        rush_yds = 0 if rush_att == 0 else int(rush_att * rng.uniform(2.5, 6.0))
        rush_td = 0 if rush_yds < 20 else rng.choice([0, 0, 1, 1, 2])
        targets = rng.choice([0, 2, 4, 5, 7, 9, 11, 13])
        rec = 0 if targets == 0 else rng.randint(max(0, targets - 4), targets)
        # Slightly higher yards-per-catch than the original (6-15 -> 7-16) so the
        # "100+ receiving yards" set task isn't dominated by the empty set.
        rec_yds = 0 if rec == 0 else int(rec * rng.uniform(7.0, 16.0))
        rec_td = 0 if rec_yds < 25 else rng.choice([0, 0, 1, 1])
        players.append({
            "name": nm,
            "rush_att": rush_att, "rush_yds": rush_yds, "rush_td": rush_td,
            "rec": rec, "rec_yds": rec_yds, "rec_td": rec_td,
        })

    # --- Team scoring DERIVED from the players (consistency invariant) --------
    off_td = sum(p["rush_td"] + p["rec_td"] for p in players)
    xp = rng.randint(0, off_td)                   # extra points made  (<= TDs)
    two_pt = rng.randint(0, max(0, off_td - xp))  # two-point conversions made
    fgs = rng.randint(0, 4)                       # field goals (independent of TDs)
    points = off_td * 6 + fgs * 3 + xp * 1 + two_pt * 2

    return {
        "players": players,
        "scoring": {
            "td": off_td, "fg": fgs, "xp": xp, "two_pt": two_pt, "points": points,
        },
    }


def render_box_score(box: dict) -> str:
    lines = ["Player | RushAtt RushYds RushTD | Rec RecYds RecTD"]
    for p in box["players"]:
        lines.append(
            f"{p['name']} | {p['rush_att']} {p['rush_yds']} {p['rush_td']} | "
            f"{p['rec']} {p['rec_yds']} {p['rec_td']}"
        )
    s = box["scoring"]
    # We deliberately do NOT print an aggregate TD count: TDs are the players'
    # rush_td + rec_td above. Printing a total would risk contradicting the rows
    # and let the model read `total_tds` off the line instead of summing.
    lines.append(
        f"TEAM SCORING (non-TD): {s['fg']} FG, {s['xp']} XP made, "
        f"{s['two_pt']} two-point conversions. "
        f"All touchdowns are the rushing/receiving TDs listed above."
    )
    return "\n".join(lines)


def generate_game_state(rng: random.Random) -> dict:
    """A late-game situation for decision-style reasoning.

    The deficit is sampled so the TD-vs-FG label is ~50/50 (before, score_diff
    was uniform over -8..-1, making "need a TD" the answer 75% of the time -- a
    model could score 75% by always answering TD without reasoning)."""
    down = rng.randint(1, 4)
    distance = rng.randint(1, 15)
    yardline = rng.randint(2, 95)          # distance from opponent end zone
    if rng.random() < 0.5:
        deficit = rng.choice([1, 2])               # FG takes the lead
    else:
        deficit = rng.choice([3, 4, 5, 6, 7, 8])   # need a TD
    score_diff = -deficit
    minutes = rng.randint(0, 2)
    seconds = rng.choice([5, 12, 25, 38, 47, 59])
    return {
        "down": down, "distance": distance, "yardline": yardline,
        "score_diff": score_diff, "minutes": minutes, "seconds": seconds,
    }


def render_game_state(gs: dict) -> str:
    ord_down = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}[gs["down"]]
    return (
        f"Situation: offense trailing by {abs(gs['score_diff'])}. "
        f"{ord_down} & {gs['distance']} at the opponent {gs['yardline']}-yard line. "
        f"{gs['minutes']}:{gs['seconds']:02d} remaining, no timeouts."
    )
