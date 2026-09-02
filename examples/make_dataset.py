#!/usr/bin/env python3
"""Regenerate ``examples/data/comparisons.jsonl``.

The dataset is committed so the demo runs without this script, but the script is
committed too so nobody has to take the data on trust. Run it and diff the output:

    python examples/make_dataset.py

Content is invented arithmetic. There is no scraped text, no licensed corpus and no
personal data in it -- it exists to exercise the intervention machinery, not to measure
anything about the world.

Four templates, chosen so that each intervention in the demo spec has a *different*
relationship to the questions:

* ``direct`` / ``direct-negated`` -- deleting negation flips the question, so the
  ``strip-negation`` arm has a large, real effect.
* ``ordinal-first`` -- "is the first number greater than the second" survives having its
  articles deleted, because reading operands left to right still gives the right answer.
* ``ordinal-second`` -- "is the second number greater than the first" does not, so
  ``strip-articles`` has a small effect concentrated on this slice.

Politeness prefixes appear on half the items and are semantically inert, which is what
makes the ``strip-politeness`` arm a genuine null rather than a small effect that
happened to miss.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

N_ITEMS = 200
SEED = 20260902

# Item counts per template. ordinal-second is the small slice that strip-articles
# damages; its size is what puts that arm near the edge of detectability at N=200.
TEMPLATE_COUNTS = {
    "direct": 70,
    "direct-negated": 70,
    "ordinal-first": 40,
    "ordinal-second": 20,
}

POLITENESS = ("", "", "Please, ", "Kindly, ")

ANSWER_INSTRUCTION = "Answer with one word: true or false."


def build(template: str, left: int, right: int, polite: str) -> tuple[str, str]:
    """Return (prompt, expected) for one item."""
    if template == "direct":
        prompt = f"{polite}Is {left} greater than {right}? {ANSWER_INSTRUCTION}"
        truth = left > right
    elif template == "direct-negated":
        prompt = f"{polite}Is {left} not greater than {right}? {ANSWER_INSTRUCTION}"
        truth = not (left > right)
    elif template == "ordinal-first":
        prompt = (
            f"{polite}Consider the numbers {left} and {right}. Is the first number "
            f"greater than the second number? {ANSWER_INSTRUCTION}"
        )
        truth = left > right
    elif template == "ordinal-second":
        prompt = (
            f"{polite}Consider the numbers {left} and {right}. Is the second number "
            f"greater than the first number? {ANSWER_INSTRUCTION}"
        )
        truth = right > left
    else:  # pragma: no cover - guarded by TEMPLATE_COUNTS
        raise ValueError(f"unknown template {template!r}")

    return prompt, "true" if truth else "false"


def main() -> None:
    rng = random.Random(SEED)
    templates: list[str] = []
    for name, count in TEMPLATE_COUNTS.items():
        templates.extend([name] * count)
    assert len(templates) == N_ITEMS, f"template counts sum to {len(templates)}, want {N_ITEMS}"
    rng.shuffle(templates)

    used: set[tuple[int, int]] = set()
    records = []
    for index, template in enumerate(templates):
        # Distinct operand pairs, so every item has its own difficulty draw in the mock
        # provider (which keys difficulty off the operands, not the prompt text).
        while True:
            left = rng.randint(1, 99)
            right = rng.randint(1, 99)
            if left != right and (left, right) not in used:
                used.add((left, right))
                break
        polite = rng.choice(POLITENESS)
        prompt, expected = build(template, left, right, polite)
        records.append(
            {
                "id": f"cmp.{index:04d}",
                "prompt": prompt,
                "expected": expected,
                "template": template,
            }
        )

    target = Path(__file__).parent / "data" / "comparisons.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    counts = {name: sum(1 for r in records if r["template"] == name) for name in TEMPLATE_COUNTS}
    print(f"wrote {len(records)} items to {target}")
    print(f"  templates: {counts}")
    print(f"  expected true: {sum(1 for r in records if r['expected'] == 'true')}")


if __name__ == "__main__":
    main()
