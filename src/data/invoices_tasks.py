"""Invoice task templates -> verifiable (context, question, answer, answer_type).

This is the invoices twin of data/tasks.py. It exposes the SAME public surface
-- a `Sample` dataclass, a `KIND_DEPTH` map, an `ALL_TASKS` list, and
`sample_one(rng)` -- so build_dataset.py can swap domains with a single flag and
the UNCHANGED rewards in src/rewards/verifiers.py score every row.

answer_type drives how the reward function checks correctness (identical
contract to the football domain):
  - "numeric"  : single number, compared with a tiny tolerance. Money answers
                 are rendered as plain dollars with two decimals (no '$', no
                 commas), e.g. 1234.50, matching the system prompt's numeric rule
                 and parsing exactly under the reward's number parser.
  - "name"     : a single line-item description; normalized string compare. The
                 generator guarantees a UNIQUE LAST WORD per item, so a last-word
                 answer ("Mouse" for "Wireless Mouse") is unambiguous, exactly as
                 a last name is for a player.
  - "set"      : a set of item descriptions, order-insensitive.
  - "decision" : a single token from a 2-word vocab (OVER / UNDER), the budget
                 analog of the football TD / FG decision.

`depth` is the compositional depth (how many rows/ops the answer requires) and
travels with each row so eval can group by it -- same Q3 taxonomy as football.
"""

import random
from dataclasses import dataclass

from data.invoices_generators import generate_invoice, render_invoice


@dataclass
class Sample:
    context: str
    question: str
    answer: str
    answer_type: str
    kind: str
    depth: int  # 1=low (single comparison) ... 4=high (filter/aggregate all rows)


# Compositional-depth labels (low -> high). Single source of truth shared by the
# dataset and the eval grouping, mirroring KIND_DEPTH in data/tasks.py. The
# depths are chosen to line up with the football ladder so cross-domain
# depth comparisons are apples-to-apples:
#   1  one comparison / commit-early decision      (football: td_or_fg)
#   2  locate one row, combine its fields          (football: scrimmage_total)
#   3  per-row reduce then argmax, or sum a column  (football: most_scrimmage / total_tds)
#   4  combine an aggregate line, or filter all rows (football: team_points / hundred_yd_rec)
KIND_DEPTH = {
    "over_budget": 1,        # compare grand total to a stated budget; OVER/UNDER
    "line_item_total": 2,    # locate one item, read (or qty * unit_price) its line total
    "item_count": 3,         # count the line items (reduce over all rows)
    "highest_line_item": 3,  # per-row line totals, then argmax
    "grand_total": 4,        # combine subtotal + tax + shipping (composite-arithmetic wall)
    "items_over_amount": 4,  # filter all rows by a line-total threshold -> set
}


def _money_answer(cents: int) -> str:
    """Numeric money answer: plain dollars, two decimals, no '$' and no commas
    (matches the system prompt's numeric rule and the printed LineTotal). Parses
    exactly under the reward's _parse_number."""
    return f"{cents // 100}.{cents % 100:02d}"


# --- invoice tasks ---------------------------------------------------------

def task_line_item_total(rng: random.Random) -> Sample:
    """Line total for one named item (locate the row, report qty * unit price).

    depth 2: the football scrimmage_total analog -- find one row, combine its
    fields. The line total is printed, but the model must still pick the right
    row (and can verify via qty * unit price)."""
    inv = generate_invoice(rng)
    it = rng.choice(inv["items"])
    return Sample(
        context=render_invoice(inv),
        question=f"What is the line total for \"{it['description']}\" (in dollars)?",
        answer=_money_answer(it["line_total"]), answer_type="numeric",
        kind="line_item_total", depth=KIND_DEPTH["line_item_total"],
    )


def task_grand_total(rng: random.Random) -> Sample:
    """Grand total including tax and shipping (subtotal + tax + shipping).

    depth 4: the composite-arithmetic WALL, the invoices analog of team_points.
    Requires combining three labeled money lines rather than copying one number."""
    inv = generate_invoice(rng)
    return Sample(
        context=render_invoice(inv),
        question="What is the grand total of this invoice including tax and shipping (in dollars)?",
        answer=_money_answer(inv["grand_total"]), answer_type="numeric",
        kind="grand_total", depth=KIND_DEPTH["grand_total"],
    )


def task_highest_line_item(rng: random.Random) -> Sample:
    """Which line item has the highest line total (argmax over the table).

    depth 3: the most_scrimmage analog. On a tie for the top line total we
    RE-DRAW a fresh invoice rather than silently changing the sample's kind,
    keeping the realized kind distribution honest (matches task_most_scrimmage)."""
    inv = generate_invoice(rng)
    for _ in range(64):  # bounded retries; ties are rare
        totals = sorted((it["line_total"] for it in inv["items"]), reverse=True)
        if len(totals) < 2 or totals[0] != totals[1]:
            break
        inv = generate_invoice(rng)
    best = max(inv["items"], key=lambda it: (it["line_total"], it["description"]))
    return Sample(
        context=render_invoice(inv),
        question="Which line item has the highest line total?",
        answer=best["description"], answer_type="name",
        kind="highest_line_item", depth=KIND_DEPTH["highest_line_item"],
    )


def task_items_over_amount(rng: random.Random) -> Sample:
    """Set membership: every line item whose line total is >= a threshold $X.

    depth 4: the hundred_yd_rec analog (filter all rows -> set). The threshold is
    drawn FROM the actual line-total distribution (between the 2nd-smallest and
    the largest) so the answer is rarely the empty or the full set, keeping the
    task informative -- the same spirit as the football 100-yard tuning."""
    inv = generate_invoice(rng)
    totals = sorted(it["line_total"] for it in inv["items"])
    # Pick a threshold strictly above the smallest and at/below the largest so at
    # least one item qualifies and at least one is excluded (when possible).
    if len(totals) >= 2 and totals[0] != totals[-1]:
        low = totals[0] + 1
        high = totals[-1]
        threshold = rng.randint(low, high)
    else:
        threshold = totals[-1] if totals else 0
    qualifiers = [it["description"] for it in inv["items"] if it["line_total"] >= threshold]
    answer = ", ".join(sorted(qualifiers)) if qualifiers else "none"
    dollars = _money_answer(threshold)
    return Sample(
        context=render_invoice(inv),
        question=(
            f"List every line item whose line total is ${dollars} or more "
            f"(comma-separated, or 'none')."
        ),
        answer=answer, answer_type="set",
        kind="items_over_amount", depth=KIND_DEPTH["items_over_amount"],
    )


def task_over_budget(rng: random.Random) -> Sample:
    """Decision: is the invoice's grand total OVER or UNDER a stated budget?

    depth 1: the td_or_fg analog -- a single comparison, commit-early. The budget
    is sampled ~50/50 above and below the grand total so a model can't win by
    always answering one token (mirrors the TD/FG ~50/50 balancing). We avoid the
    exact-tie case so OVER/UNDER is always well-defined."""
    inv = generate_invoice(rng)
    gt = inv["grand_total"]
    # Budget in whole dollars, offset from the grand total by 1..20%, half the
    # time below (=> OVER) and half above (=> UNDER). Never equal.
    pct = rng.randint(1, 20)
    delta = max(100, gt * pct // 100)  # at least $1 so it never lands on a tie
    if rng.random() < 0.5:
        budget = gt - delta            # grand total exceeds budget -> OVER
        answer = "OVER"
    else:
        budget = gt + delta            # grand total under budget -> UNDER
        answer = "UNDER"
    budget = max(100, (budget // 100) * 100)  # whole dollars, positive
    # Re-derive the label in case the dollar-rounding nudged us across the line
    # (keeps the answer EXACTLY consistent with the printed number).
    answer = "OVER" if gt > budget else "UNDER"
    budget_str = _money_answer(budget)
    return Sample(
        context=render_invoice(inv),
        question=(
            f"Is the grand total OVER or UNDER a budget of ${budget_str}? "
            f"Answer OVER or UNDER."
        ),
        answer=answer, answer_type="decision",
        kind="over_budget", depth=KIND_DEPTH["over_budget"],
    )


def task_item_count(rng: random.Random) -> Sample:
    """How many line items are on the invoice (reduce over all rows).

    depth 3: a count/aggregate over every row (the total_tds analog -- you must
    read the whole table, not one cell). We VARY the number of items (3..7) so
    the count is not a constant a model could memorize -- the same reason the
    football aggregates aren't printed as a single line."""
    inv = generate_invoice(rng, n_items=rng.randint(3, 7))
    return Sample(
        context=render_invoice(inv),
        question="How many distinct line items are on this invoice?",
        answer=str(len(inv["items"])), answer_type="numeric",
        kind="item_count", depth=KIND_DEPTH["item_count"],
    )


ALL_TASKS = [
    task_line_item_total,
    task_grand_total,
    task_highest_line_item,
    task_items_over_amount,
    task_over_budget,
    task_item_count,
]


def sample_one(rng: random.Random) -> Sample:
    return rng.choice(ALL_TASKS)(rng)
