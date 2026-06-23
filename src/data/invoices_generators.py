"""Synthetic invoice generators -- a SECOND structured-data domain.

Why a second domain: the whole pitch of this repo is "the pipeline is
domain-agnostic; the only file you need to change is the data layer." Football
box scores prove the idea once; invoices prove it *generalizes* -- same Sample
schema, same prompt builder, same UNCHANGED verifiable rewards. (See REVIEW.md:
the "swap the data layer" claim was previously asserted but unbacked.)

Why synthetic (same rationale as generators.py): infinite, perfectly-labeled
data with zero licensing/PII issues, and you can dial difficulty. The ground
truth is COMPUTED, not annotated, so the verifiable reward is exact by
construction. Swap in real invoices (e.g. a parsed PDF corpus) for the EVAL set
later to make a headline number credible -- keep training synthetic.

Everything is seeded for reproducibility.

Design invariants (mirrored from generators.py so the two domains behave the
same under the reward checker):
  * Money is exact in CENTS. Every line total, the subtotal, the tax, shipping,
    and the grand total are whole numbers of cents, so the numeric reward (which
    compares with a 1e-6 tolerance, i.e. effectively exact) never penalizes a
    model that does the arithmetic correctly. Tax is applied to the subtotal and
    ROUNDED to the nearest cent, exactly as a real invoice does; the grand total
    is DERIVED from subtotal + tax + shipping, so the table is internally
    consistent (the analog of team points being derived from player TDs).
  * ITEM DESCRIPTIONS HAVE A UNIQUE LAST WORD within an invoice. The name reward
    accepts a correct last-word-only answer (just as it accepts a last name for a
    player), so "Mouse" must unambiguously identify "Wireless Mouse". This makes
    every "which line item ..." answer well-posed.
  * The grand total is NOT printed pre-broken into its parts beyond the labeled
    Subtotal / Tax / Shipping lines a real invoice shows; the composite-arithmetic
    task still requires combining three numbers (the wall analog to team_points).
"""

import random

# Catalog of (description, unit_price_cents) pairs. Prices are realistic and in
# CENTS so all arithmetic stays integer-exact. The LAST WORD of every
# description is a distinct noun, which is what the name reward keys on (it
# accepts a last-word-only answer, like a player's last name), so "which item"
# questions remain unambiguous when we enforce unique last words below.
CATALOG = [
    ("Wireless Mouse", 2499),
    ("Mechanical Keyboard", 8999),
    ("Laptop Stand", 3450),
    ("USB-C Hub", 5299),
    ("Noise-Cancelling Headphones", 19999),
    ("Webcam", 6750),
    ("Monitor Arm", 11999),
    ("Desk Lamp", 4299),
    ("Ergonomic Chair", 24900),
    ("Standing Desk", 38999),
    ("Cable Organizer", 1299),
    ("Laptop Sleeve", 2999),
    ("Portable Charger", 4599),
    ("Bluetooth Speaker", 7999),
    ("Graphics Tablet", 9950),
    ("External SSD", 12499),
    ("Document Scanner", 15900),
    ("Label Printer", 8450),
    ("Office Chair", 17999),
    ("Whiteboard", 6299),
]

# Plausible business customer names; cosmetic only (not used in any answer), so
# they don't need a uniqueness invariant.
CUSTOMERS = [
    "Northwind Traders",
    "Acme Supplies",
    "Globex Corp",
    "Initech LLC",
    "Umbrella Retail",
    "Soylent Foods",
    "Stark Industries",
    "Wayne Enterprises",
    "Hooli Inc",
    "Pied Piper",
    "Vandelay Imports",
    "Wonka Logistics",
]


def _unique_last_word_items(rng: random.Random, n: int) -> list[tuple[str, int]]:
    """Pick n catalog rows whose DESCRIPTIONS HAVE UNIQUE LAST WORDS, so a
    last-word-only answer (which the name reward accepts, mirroring last-name
    matching for players) unambiguously identifies one line item."""
    chosen: list[tuple[str, int]] = []
    used_last: set[str] = set()
    pool = CATALOG[:]
    rng.shuffle(pool)
    for desc, price in pool:
        last = desc.split()[-1].lower()
        if last in used_last:
            continue
        used_last.add(last)
        chosen.append((desc, price))
        if len(chosen) == n:
            break
    return chosen


def generate_invoice(rng: random.Random, n_items: int = 5) -> dict:
    """A simplified vendor invoice: a handful of line items (description, qty,
    unit price, line total) plus subtotal, a tax rate, shipping, and a grand
    total. Line items are independent draws; the money totals are DERIVED from
    them so the invoice is internally consistent.

    All amounts are stored in integer CENTS so every total is exact."""
    items_spec = _unique_last_word_items(rng, n_items)
    items = []
    for desc, unit_price in items_spec:
        qty = rng.randint(1, 12)
        line_total = qty * unit_price  # exact in cents
        items.append(
            {
                "description": desc,
                "qty": qty,
                "unit_price": unit_price,  # cents
                "line_total": line_total,  # cents
            }
        )

    subtotal = sum(it["line_total"] for it in items)  # cents

    # Tax rate as an integer count of basis points so the math is exact and the
    # printed percent is clean (e.g. 825 bp -> 8.25%). Tax in cents is the
    # subtotal times the rate, rounded to the nearest cent (banker-free, the way
    # an invoice rounds): round(subtotal_cents * bp / 10000).
    tax_bp = rng.choice([0, 500, 625, 700, 825, 900, 1000])  # basis points
    tax = int((subtotal * tax_bp + 5000) // 10000)  # cents, rounded half-up

    # Shipping is a flat handling fee in whole dollars (so always whole cents).
    shipping = rng.choice([0, 1500, 2500, 4000, 7500]) if rng.random() < 0.85 else 0

    grand_total = subtotal + tax + shipping  # cents, DERIVED

    return {
        "customer": rng.choice(CUSTOMERS),
        "invoice_no": f"INV-{rng.randint(10000, 99999)}",
        "items": items,
        "subtotal": subtotal,
        "tax_bp": tax_bp,
        "tax": tax,
        "shipping": shipping,
        "grand_total": grand_total,
    }


def _money(cents: int) -> str:
    """Render integer cents as a dollar amount with exactly two decimals."""
    return f"${cents // 100}.{cents % 100:02d}"


def render_invoice(inv: dict) -> str:
    """Compact text table, in the spirit of render_box_score: a header row, one
    line per item, then the labeled money lines a real invoice shows. Unit price
    and line total are printed as dollars; quantities are integers."""
    lines = [
        f"INVOICE {inv['invoice_no']} -- Bill to: {inv['customer']}",
        "Item | Qty | UnitPrice | LineTotal",
    ]
    for it in inv["items"]:
        lines.append(
            f"{it['description']} | {it['qty']} | "
            f"{_money(it['unit_price'])} | {_money(it['line_total'])}"
        )
    # Labeled totals, exactly as an invoice prints them. The grand total is the
    # sum of these three lines; we print the parts (a real invoice does) but the
    # composite task still requires the model to add them.
    rate = f"{inv['tax_bp'] // 100}.{inv['tax_bp'] % 100:02d}%"
    lines.append(f"Subtotal: {_money(inv['subtotal'])}")
    lines.append(f"Tax ({rate}): {_money(inv['tax'])}")
    lines.append(f"Shipping: {_money(inv['shipping'])}")
    lines.append(f"Grand Total: {_money(inv['grand_total'])}")
    return "\n".join(lines)
