#!/usr/bin/env python3
"""What squashing unresolved trials into failures actually costs.

    python examples/unresolved_trials.py

`passed` is a boolean, so a harness that hits a truncated trace or a provider error has
nowhere honest to put it. The tempting move is `passed: false` -- it is one character of
code and the run keeps going. This script measures what that character costs.

It takes the results directory shipped in `tests/fixtures/foreign_tool_results/`, marks a
deterministic tenth of the trials unresolved, and reads the same run two ways:

* **honest** -- those trials carry `verdict: "inconclusive"` (CXS 0.1.1) and errorbars
  skips them, so they enter no rate and no denominator;
* **folded** -- the same trials carry `passed: false`, the squash.

Both numbers come out of `errorbars analyze`. Neither is asserted by hand.

The fixture is synthetic and seeded (see `tests/fixtures/make_foreign_fixture.py`); it is
not a measurement of any model. What is being demonstrated is arithmetic, and that part
is real.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from errorbars.ingest import load_run
from errorbars.stats.effects import analyse_arms

FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "foreign_tool_results"

# Every tenth outcome. Deterministic, so the numbers below are reproducible.
UNRESOLVED_EVERY = 10


def build(destination: Path, *, fold_into_failures: bool) -> Path:
    """Copy the fixture, marking every Nth outcome unresolved -- or squashed."""
    shutil.copytree(FIXTURE, destination)
    path = destination / "outcomes.jsonl"

    rewritten = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        record = json.loads(line)
        if index % UNRESOLVED_EVERY == 0:
            if fold_into_failures:
                record["passed"] = False
            else:
                record.pop("passed", None)
                record["verdict"] = "inconclusive"
        rewritten.append(json.dumps(record))

    path.write_text("\n".join(rewritten) + "\n", encoding="utf-8")
    return destination


def read(directory: Path) -> tuple[float, float, float, float, str, int]:
    run = load_run(directory)
    analysis = analyse_arms(
        run.control, run.treatments, bootstrap_resamples=4000, permutations=4000, seed=0
    )
    effect = next(e for e in analysis.effects if e.intervention_id == "shuffle-context")
    return (
        effect.control_rate,
        effect.effect,
        effect.ci_low,
        effect.ci_high,
        effect.verdict,
        run.skipped_unresolved,
    )


def main() -> None:
    with tempfile.TemporaryDirectory() as workspace:
        root = Path(workspace)
        honest = read(build(root / "honest", fold_into_failures=False))
        folded = read(build(root / "folded", fold_into_failures=True))

    print(f"One tenth of the trials never resolved.  ({honest[5]} of them, skipped honestly)\n")
    print(f"{'':<10}{'control rate':>14}{'effect':>10}{'95% CI':>18}{'verdict':>15}")
    print("-" * 67)
    for label, (rate, effect, low, high, verdict, _) in (("honest", honest), ("folded", folded)):
        interval = f"[{low * 100:+.1f}, {high * 100:+.1f}]"
        print(f"{label:<10}{rate:>13.1%}{effect * 100:>9.1f}pp{interval:>18}{verdict:>15}")

    print()
    print(f"  control rate moved by {(honest[0] - folded[0]) * 100:.1f} percentage points")
    print(f"  effect size moved by  {(honest[1] - folded[1]) * 100:.1f} percentage points")
    print()
    print("  Folding unresolved trials into the denominator as failures does not add")
    print("  noise -- it adds bias, in a known direction, to every arm at once.")


if __name__ == "__main__":
    main()
