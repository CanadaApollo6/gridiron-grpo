"""Synthetic NFL data generators.

Why synthetic: it gives infinite, perfectly-labeled training data with zero
licensing/scraping issues, and lets you dial difficulty. The ground truth is
computed, not annotated, so the verifiable reward is exact by construction.

Swap in real data (nfl_data_py / nflverse, or NGS exports) for the EVAL set
later to make the headline number credible -- keep training synthetic.

Everything is seeded for reproducibility.
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
    names: set[str] = set()
    while len(names) < n:
        names.add(_name(rng))
    return list(names)


def generate_box_score(rng: random.Random, n_skill: int = 6) -> dict:
    """A simplified offensive box score for a set of skill players plus team
    scoring. All numbers are independent draws; ground truth is derived from
    these fields downstream."""
    names = _unique_names(rng, n_skill)
    players = []
    for nm in names:
        rush_att = rng.choice([0, 0, 0, 3, 6, 10, 14, 18, 22])
        rush_yds = 0 if rush_att == 0 else int(rush_att * rng.uniform(2.5, 6.0))
        rush_td = 0 if rush_yds < 20 else rng.choice([0, 0, 1, 1, 2])
        targets = rng.choice([0, 2, 4, 5, 7, 9, 11, 13])
        rec = 0 if targets == 0 else rng.randint(max(0, targets - 4), targets)
        rec_yds = 0 if rec == 0 else int(rec * rng.uniform(6.0, 15.0))
        rec_td = 0 if rec_yds < 25 else rng.choice([0, 0, 1, 1])
        players.append({
            "name": nm,
            "rush_att": rush_att, "rush_yds": rush_yds, "rush_td": rush_td,
            "rec": rec, "rec_yds": rec_yds, "rec_td": rec_td,
        })

    # Team scoring drives -> points (verifiable arithmetic)
    tds = rng.randint(1, 5)
    fgs = rng.randint(0, 4)
    xp = rng.randint(0, tds)  # extra points made (<= TDs)
    two_pt = rng.randint(0, max(0, tds - xp))
    points = tds * 6 + fgs * 3 + xp * 1 + two_pt * 2

    return {
        "players": players,
        "scoring": {"td": tds, "fg": fgs, "xp": xp, "two_pt": two_pt, "points": points},
    }


def render_box_score(box: dict) -> str:
    lines = ["Player | RushAtt RushYds RushTD | Rec RecYds RecTD"]
    for p in box["players"]:
        lines.append(
            f"{p['name']} | {p['rush_att']} {p['rush_yds']} {p['rush_td']} | "
            f"{p['rec']} {p['rec_yds']} {p['rec_td']}"
        )
    s = box["scoring"]
    lines.append(
        f"TEAM SCORING: {s['td']} TD, {s['fg']} FG, {s['xp']} XP made, "
        f"{s['two_pt']} two-point conversions"
    )
    return "\n".join(lines)


def generate_game_state(rng: random.Random) -> dict:
    """A late-game situation for decision-style reasoning."""
    down = rng.randint(1, 4)
    distance = rng.randint(1, 15)
    yardline = rng.randint(2, 95)          # distance from opponent end zone
    score_diff = rng.choice([-8, -7, -6, -5, -4, -3, -2, -1])  # offense trailing
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
