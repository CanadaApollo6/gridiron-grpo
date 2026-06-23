"""Build a REAL-DATA held-out eval set from nflverse weekly player stats.

Why: the model TRAINS on synthetic, perfectly-labeled box scores (see
src/data/generators.py). To make the headline number externally credible we
EVAL the SAME verifiable tasks on REAL NFL stat lines -- "train on synthetic
verifiable tasks, evaluate on real games." This script pulls real weekly skill
stats, folds them into per-team / per-game box scores rendered in the EXACT
`render_box_score` format, and emits eval rows in the SAME JSONL schema as
src/data/build_dataset.py, consumable by src/eval/evaluate.py unchanged.

Format guarantee: we import and call the real `render_box_score` so a real row
is byte-for-byte indistinguishable in layout from a synthetic one. Stats are
cast to int to match the synthetic integer columns; names are reformatted to
the synthetic "F. Last" convention.

Tasks emitted (matching src/data/tasks.py question strings + answer logic):
  - scrimmage_total  (numeric)  rush_yds + rec_yds for one named player
  - total_tds        (numeric)  sum rush_td + rec_td across the listed players
  - most_scrimmage   (name)     argmax yards-from-scrimmage over the listed players
  - hundred_yd_rec   (set)      players with >= 100 receiving yards

Deliberately SKIPPED:
  - td_or_fg    -- needs a synthetic game-state (down/distance/clock); no real
                   analogue in weekly box data.
  - team_points -- NOT reconstructable from skill-player weekly rows. The
                   synthetic scoring model is closed-world ("all TDs are the
                   rushing/receiving TDs listed, plus a FG/XP/2pt line"); a real
                   team's points also include passing TDs, defensive/ST TDs,
                   safeties, kicker FGs/XPs -- none of which appear in the
                   weekly skill table. We therefore omit team_points rather than
                   emit a wrong ground truth. (The four tasks we DO emit never
                   read the scoring line, so the box stays well-posed for them.)

Determinism: a single integer --seed drives every random choice (which player a
scrimmage_total asks about, tie re-rolls). Given the same --years/--week/--seed
and the same nflverse release, the output JSONL is byte-stable. The nflverse
parquet pull is cached to disk so reruns are offline and fast.

Usage:
  .venv/bin/python3 src/data/build_real_eval.py \
      --years 2023 --week 1 --out data_out/eval_real.jsonl --seed 7
  # multiple weeks:
  .venv/bin/python3 src/data/build_real_eval.py --years 2023 --week 1-4 ...
"""

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.generators import render_box_score  # noqa: E402  (EXACT format reuse)
from data.tasks import KIND_DEPTH  # noqa: E402  (single source of truth for depth)
from prompts import build_prompt  # noqa: E402

# Skill positions whose rushing/receiving lines populate the offensive box.
# (Matches what the synthetic generator models: RB/WR/TE/QB/FB touches.)
SKILL_POSITIONS = {"QB", "RB", "WR", "TE", "FB"}

# nflverse weekly column -> synthetic box field.
COLMAP = {
    "rush_att": "carries",
    "rush_yds": "rushing_yards",
    "rush_td": "rushing_tds",
    "rec": "receptions",
    "rec_yds": "receiving_yards",
    "rec_td": "receiving_tds",
}


# ---------------------------------------------------------------------------
# nflverse pull (cached to disk so reruns are offline + deterministic).
# ---------------------------------------------------------------------------
def load_weekly(years: list[int], cache_dir: Path):
    """Return the nflverse weekly player-stats DataFrame for `years`, cached to
    parquet under cache_dir. Network is only hit on a cache miss."""
    import pandas as pd

    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    missing = []
    for y in years:
        p = cache_dir / f"weekly_{y}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
        else:
            missing.append((y, p))

    if missing:
        import nfl_data_py as nfl

        for y, p in missing:
            # downcast=False keeps int columns as ints (carries/tds/receptions),
            # which we rely on; floats (yards) are cast to int at render time.
            df = nfl.import_weekly_data([y], downcast=False)
            df.to_parquet(p)
            frames.append(df)
            print(f"[cache] fetched + wrote {p} ({len(df)} rows)")
    else:
        print(f"[cache] hit for years={years} under {cache_dir}")

    return pd.concat(frames, ignore_index=True)


def parse_week_arg(week: str) -> list[int]:
    """'1' -> [1]; '1-4' -> [1,2,3,4]; '1,3,5' -> [1,3,5]."""
    out: list[int] = []
    for part in str(week).split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


# ---------------------------------------------------------------------------
# Real stat line -> synthetic-format box dict.
# ---------------------------------------------------------------------------
def _fmt_name(player_name: str, display_name: str) -> str:
    """Reformat an nflverse name to the synthetic 'F. Last' convention.

    nflverse `player_name` is 'A.Rodgers' (initial dot lastname, no space). We
    want 'A. Rodgers'. Prefer that compact field; fall back to building from the
    display name ('Aaron Rodgers' -> 'A. Rodgers'). Last name may itself contain
    a space ("Amon-Ra St. Brown" -> display fallback gives 'A. St. Brown')."""
    pn = (player_name or "").strip()
    if "." in pn:
        initial, last = pn.split(".", 1)
        initial, last = initial.strip(), last.strip()
        if initial and last:
            return f"{initial}. {last}"
    dn = (display_name or "").strip()
    if dn:
        parts = dn.split()
        if len(parts) >= 2:
            return f"{parts[0][0]}. {' '.join(parts[1:])}"
    return pn or dn or "Unknown"


def _last_name(name: str) -> str:
    return name.split()[-1].lower() if name else ""


def _to_int(v) -> int:
    try:
        if v is None:
            return 0
        import math

        f = float(v)
        if math.isnan(f):
            return 0
        return int(round(f))
    except (TypeError, ValueError):
        return 0


def build_box(group_df, n_skill: int) -> dict | None:
    """Fold one team's per-game skill rows into a synthetic-format box dict.

    Picks the top `n_skill` players by usage (carries + targets, scrimmage yards
    as tiebreak), enforces UNIQUE LAST NAMES within the box (the synthetic
    invariant that makes name/set answers well-posed and last-name-only answers
    unambiguous), and casts every stat to int. Returns None if fewer than 2
    distinct-last-name skill players are available (box too thin to be useful)."""
    df = group_df[group_df["position"].isin(SKILL_POSITIONS)].copy()
    if df.empty:
        return None

    # usage rank: total touches, then scrimmage yards as a deterministic tiebreak
    df["_carries"] = df["carries"].fillna(0).astype(float)
    df["_targets"] = df["targets"].fillna(0).astype(float)
    df["_ry"] = df["rushing_yards"].fillna(0).astype(float)
    df["_recy"] = df["receiving_yards"].fillna(0).astype(float)
    df["_touches"] = df["_carries"] + df["_targets"]
    df["_scrim"] = df["_ry"] + df["_recy"]
    # drop pure zero-usage rows (0 touches AND 0 scrimmage yards) -- they'd add
    # noise rows that no synthetic box would contain.
    df = df[(df["_touches"] > 0) | (df["_scrim"] > 0)]
    if df.empty:
        return None
    # stable, deterministic order: touches desc, scrim desc, name asc
    df = df.sort_values(["_touches", "_scrim", "player_name"], ascending=[False, False, True])

    players = []
    used_last: set[str] = set()
    for _, r in df.iterrows():
        if len(players) >= n_skill:
            break
        name = _fmt_name(r.get("player_name", ""), r.get("player_display_name", ""))
        last = _last_name(name)
        if not last or last in used_last:
            continue  # enforce unique last names within the box
        used_last.add(last)
        players.append(
            {
                "name": name,
                "rush_att": _to_int(r.get(COLMAP["rush_att"])),
                "rush_yds": _to_int(r.get(COLMAP["rush_yds"])),
                "rush_td": _to_int(r.get(COLMAP["rush_td"])),
                "rec": _to_int(r.get(COLMAP["rec"])),
                "rec_yds": _to_int(r.get(COLMAP["rec_yds"])),
                "rec_td": _to_int(r.get(COLMAP["rec_td"])),
            }
        )

    if len(players) < 2:
        return None

    # Scoring line: rush+rec TDs of the listed players, plus the 2-pt
    # conversions we CAN read from the weekly table. FG/XP are NOT in the
    # weekly skill data, so they are left at 0 (and `points`/team_points is
    # intentionally NOT emitted as a task -- see module docstring). The line is
    # rendered only for visual parity; no emitted task reads it.
    off_td = sum(p["rush_td"] + p["rec_td"] for p in players)
    two_pt = 0
    for _, r in df.head(len(players)).iterrows():
        for c in (
            "rushing_2pt_conversions",
            "receiving_2pt_conversions",
            "passing_2pt_conversions",
        ):
            two_pt += _to_int(r.get(c))
    scoring = {"td": off_td, "fg": 0, "xp": 0, "two_pt": two_pt, "points": off_td * 6 + two_pt * 2}
    return {"players": players, "scoring": scoring}


# ---------------------------------------------------------------------------
# Box -> task rows. Question strings + answer logic mirror src/data/tasks.py
# EXACTLY (verified against the synthetic task fns).
# ---------------------------------------------------------------------------
def _row(context: str, question: str, answer: str, answer_type: str, kind: str) -> dict:
    return {
        "prompt": build_prompt(context, question),
        "ground_truth": answer,
        "answer_type": answer_type,
        "kind": kind,
        "depth": KIND_DEPTH[kind],
    }


def tasks_for_box(box: dict, rng: random.Random) -> list[dict]:
    """Generate the 4 box-derived verifiable rows for one real box.

    Mirrors task_scrimmage_total / task_total_touchdowns / task_most_scrimmage /
    task_hundred_yard_receivers from src/data/tasks.py (same question strings,
    same answer computations)."""
    context = render_box_score(box)
    players = box["players"]
    rows: list[dict] = []

    # scrimmage_total -- pick one player (seeded)
    p = rng.choice(players)
    total = p["rush_yds"] + p["rec_yds"]
    rows.append(
        _row(
            context,
            f"How many total yards from scrimmage (rushing + receiving) did {p['name']} have?",
            str(total),
            "numeric",
            "scrimmage_total",
        )
    )

    # total_tds -- sum rush+rec TDs across the listed players
    tds = sum(pl["rush_td"] + pl["rec_td"] for pl in players)
    rows.append(
        _row(
            context,
            "How many total touchdowns (rushing + receiving) did these players score combined?",
            str(tds),
            "numeric",
            "total_tds",
        )
    )

    # most_scrimmage -- argmax yards-from-scrimmage; skip on a tie at the top
    # (the synthetic task re-draws; we have a fixed box, so we just skip to keep
    # the answer well-posed -- a tie has no unique correct name).
    totals = sorted((pl["rush_yds"] + pl["rec_yds"] for pl in players), reverse=True)
    if len(totals) >= 2 and totals[0] != totals[1]:
        best = max(players, key=lambda pl: (pl["rush_yds"] + pl["rec_yds"], pl["name"]))
        rows.append(
            _row(
                context,
                "Which player had the most total yards from scrimmage?",
                best["name"],
                "name",
                "most_scrimmage",
            )
        )

    # hundred_yd_rec -- set of players with >= 100 receiving yards
    qualifiers = [pl["name"] for pl in players if pl["rec_yds"] >= 100]
    answer = ", ".join(sorted(qualifiers)) if qualifiers else "none"
    rows.append(
        _row(
            context,
            "List every player with 100 or more receiving yards (comma-separated, or 'none').",
            answer,
            "set",
            "hundred_yd_rec",
        )
    )

    return rows


# ---------------------------------------------------------------------------
def build_rows(weekly_df, season_types: list[str], weeks: list[int], n_skill: int, seed: int):
    """Iterate (season, week, team) games in a deterministic order and emit task
    rows. One RNG, advanced per box, so the whole file is reproducible."""
    rng = random.Random(seed)
    df = weekly_df[weekly_df["season_type"].isin(season_types)].copy()
    if weeks:
        df = df[df["week"].isin(weeks)]

    rows: list[dict] = []
    n_boxes = 0
    n_skipped = 0
    # deterministic game order
    keys = (
        df[["season", "week", "recent_team", "opponent_team"]]
        .drop_duplicates()
        .sort_values(["season", "week", "recent_team"])
    )
    for _, k in keys.iterrows():
        g = df[
            (df["season"] == k["season"])
            & (df["week"] == k["week"])
            & (df["recent_team"] == k["recent_team"])
            & (df["opponent_team"] == k["opponent_team"])
        ]
        box = build_box(g, n_skill=n_skill)
        if box is None:
            n_skipped += 1
            continue
        n_boxes += 1
        rows.extend(tasks_for_box(box, rng))
    return rows, n_boxes, n_skipped


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--years", type=str, default="2023", help="season(s): '2023' or '2022,2023' or '2021-2023'"
    )
    ap.add_argument(
        "--week", type=str, default="1", help="week(s) within season: '1', '1-4', or '1,3,5'"
    )
    ap.add_argument(
        "--season_type",
        type=str,
        default="REG",
        choices=["REG", "POST", "ALL"],
        help="which season type(s) to include",
    )
    ap.add_argument(
        "--n_skill",
        type=int,
        default=6,
        help="top-N skill players per team-game (matches synthetic default 6)",
    )
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="data_out/eval_real.jsonl")
    ap.add_argument("--cache_dir", type=str, default="data_out/nflverse_cache")
    ap.add_argument(
        "--limit", type=int, default=0, help="if >0, keep only the first N rows (quick smoke build)"
    )
    args = ap.parse_args()

    years = parse_week_arg(args.years)  # same int-list parser works for years
    weeks = parse_week_arg(args.week)
    season_types = ["REG", "POST"] if args.season_type == "ALL" else [args.season_type]

    weekly = load_weekly(years, Path(args.cache_dir))
    rows, n_boxes, n_skipped = build_rows(
        weekly, season_types, weeks, n_skill=args.n_skill, seed=args.seed
    )

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out = Path(args.out)
    write_jsonl(rows, out)

    dist = Counter(r["kind"] for r in rows)
    print(f"\nwrote {len(rows)} REAL eval rows to {out}")
    print(
        f"  years={years} weeks={weeks} season_type={season_types} "
        f"n_skill={args.n_skill} seed={args.seed}"
    )
    print(f"  boxes used={n_boxes}, team-games skipped (too thin)={n_skipped}")
    print(f"  kind distribution: {dict(dist)}")
    print(
        "  note: team_points and td_or_fg are intentionally omitted "
        "(not reconstructable from weekly skill data / no game-state)."
    )


if __name__ == "__main__":
    main()
