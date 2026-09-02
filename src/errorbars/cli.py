"""Command line interface. argparse only -- no click, no typer, nothing to install.

``errorbars power`` is the flagship and is deliberately the cheapest thing in the
package to run: pure arithmetic, no network, no key, no config file, no results
directory. If a user only ever runs one errorbars command, it should be that one, and it
should work three seconds after `uvx errorbars power`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .cache import ResponseCache, default_cache_dir
from .errors import ErrorbarsError
from .grid import estimate_budget
from .ingest import load_run
from .report import render_analysis_table, render_power, write_report_json, write_report_md
from .spec import load_dataset, load_spec
from .stats.effects import analyse_arms
from .stats.power import minimum_detectable_effect, power_analysis

__all__ = ["build_parser", "main"]

_SENSITIVITY_PAIR_CORRS = (0.0, 0.3, 0.6, 0.9)


def _percent_or_proportion(raw: str) -> float:
    """Accept ``0.02``, ``2%`` or ``2pp`` and return a proportion.

    People write effect sizes both ways and getting it wrong by 100x silently is the
    kind of mistake this whole tool exists to prevent.
    """
    text = raw.strip().lower()
    if text.endswith("pp"):
        return float(text[:-2]) / 100.0
    if text.endswith("%"):
        return float(text[:-1]) / 100.0
    return float(text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="errorbars",
        description="The statistics your eval harness doesn't do.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  errorbars power --items 200 --repeats 5 --baseline 0.80\n"
            "  errorbars power --items 200 --baseline 0.80 --detect 1pp\n"
            "  errorbars analyze results/my-experiment/\n"
            "  errorbars run examples/negation-sensitivity.yaml --dry-run\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"errorbars {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    power = subparsers.add_parser(
        "power",
        help="what a run of this size can detect (no network, no API key)",
        description=(
            "Minimum detectable effect and required N for a paired eval design. "
            "Everything is computed from the arguments; nothing is looked up or called."
        ),
    )
    power.add_argument("--items", type=int, default=200, help="items per arm (default: 200)")
    power.add_argument("--repeats", type=int, default=1, help="repeats per item (default: 1)")
    power.add_argument(
        "--baseline",
        type=_percent_or_proportion,
        default=0.80,
        help="control-arm success rate, e.g. 0.80 or 80%% (default: 0.80)",
    )
    power.add_argument(
        "--icc",
        type=float,
        default=0.5,
        help=(
            "correlation of repeats within an item, 0-1 (default: 0.5). 1.0 means "
            "repeats add nothing. Measure yours with `errorbars analyze`."
        ),
    )
    power.add_argument(
        "--pair-corr",
        type=float,
        default=0.0,
        help=(
            "correlation of item difficulty across arms, 0-1 (default: 0.0, which "
            "assumes pairing buys nothing, so the answer is never optimistic)"
        ),
    )
    power.add_argument(
        "--alpha", type=float, default=0.05, help="significance level (default: 0.05)"
    )
    power.add_argument("--power", type=float, default=0.80, help="target power (default: 0.80)")
    power.add_argument(
        "--detect",
        type=_percent_or_proportion,
        default=None,
        help="also report the items needed to detect this effect, e.g. 1pp",
    )
    power.add_argument(
        "--no-sensitivity",
        action="store_true",
        help="omit the pairing-assumption sensitivity block",
    )

    analyze = subparsers.add_parser(
        "analyze",
        help="statistics over a CXS results directory, whoever produced it",
        description=(
            "Paired effects with bootstrap intervals, a permutation test, Holm "
            "multiplicity correction, and a significant/null/underpowered verdict."
        ),
    )
    analyze.add_argument("results_dir", help="directory containing manifest.json and trials.jsonl")
    analyze.add_argument("--model", default=None, help="which model to analyse, as provider:model")
    analyze.add_argument("--control", default=None, help="which intervention is the control arm")
    analyze.add_argument(
        "--alpha", type=float, default=0.05, help="significance level (default: 0.05)"
    )
    analyze.add_argument(
        "--power", type=float, default=0.80, help="target power for the MDE (default: 0.80)"
    )
    analyze.add_argument(
        "--resamples", type=int, default=10_000, help="bootstrap resamples (default: 10000)"
    )
    analyze.add_argument(
        "--permutations", type=int, default=10_000, help="permutations (default: 10000)"
    )
    analyze.add_argument("--seed", type=int, default=0, help="RNG seed (default: 0, so runs agree)")
    analyze.add_argument(
        "--sesoi",
        type=_percent_or_proportion,
        default=None,
        help=(
            "smallest effect size of interest, e.g. 2pp. Defaults to the run's MDE. "
            "Passing your own makes the null verdict a real equivalence claim."
        ),
    )
    analyze.add_argument(
        "--simultaneous-ci",
        action="store_true",
        help=(
            "build Bonferroni-adjusted simultaneous intervals at alpha/m instead of "
            "per-comparison intervals, so all m of them hold together and the interval "
            "column controls the same family-wise error rate as the Holm p-values"
        ),
    )
    analyze.add_argument(
        "--json", action="store_true", help="write report.json into the results dir"
    )
    analyze.add_argument("--md", action="store_true", help="write report.md into the results dir")

    run = subparsers.add_parser(
        "run",
        help="execute an experiment.yaml and write a CXS results directory",
        description=(
            "A small intervention-grid runner with a mandatory control arm. Deliberately "
            "less capable than promptfoo; it exists so the control/intervention primitive "
            "has a first-class expression."
        ),
    )
    run.add_argument("spec", help="path to experiment.yaml")
    run.add_argument("--out", default=None, help="results directory (default: results/<name>)")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact model-call count and estimated tokens, then stop",
    )
    run.add_argument(
        "--max-calls", type=int, default=None, help="hard stop after this many model calls"
    )
    run.add_argument(
        "--cache-dir", default=None, help=f"response cache (default: {default_cache_dir()})"
    )
    run.add_argument("--no-cache", action="store_true", help="disable the response cache")
    run.add_argument("--no-resume", action="store_true", help="ignore existing trials.jsonl")
    run.add_argument(
        "--analyze", action="store_true", help="run `analyze` on the results afterwards"
    )

    return parser


def _cmd_power(args: argparse.Namespace) -> int:
    result = power_analysis(
        args.baseline,
        args.items,
        repeats=args.repeats,
        icc=args.icc,
        pair_corr=args.pair_corr,
        alpha=args.alpha,
        power=args.power,
        target_effect=args.detect,
    )
    sensitivity = None
    if not args.no_sensitivity:
        sensitivity = [
            (
                pair_corr,
                minimum_detectable_effect(
                    args.baseline,
                    args.items,
                    alpha=args.alpha,
                    power=args.power,
                    repeats=args.repeats,
                    icc=args.icc,
                    pair_corr=pair_corr,
                ),
            )
            for pair_corr in _SENSITIVITY_PAIR_CORRS
        ]
    print(render_power(result, sensitivity=sensitivity))
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    run = load_run(args.results_dir, model=args.model, control=args.control)
    analysis = analyse_arms(
        run.control,
        run.treatments,
        alpha=args.alpha,
        power=args.power,
        bootstrap_resamples=args.resamples,
        permutations=args.permutations,
        seed=args.seed,
        sesoi=args.sesoi,
        simultaneous_ci=args.simultaneous_ci,
    )
    print(render_analysis_table(run, analysis))

    if args.json:
        path = write_report_json(run.results_dir / "report.json", run, analysis)
        print(f"\nwrote {path}")
    if args.md:
        path = write_report_md(run.results_dir / "report.md", run, analysis)
        print(f"wrote {path}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .runner import run_experiment  # imported lazily so `power` stays fast to start

    spec, warnings = load_spec(args.spec)
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    items = load_dataset(spec.dataset_path)
    budget = estimate_budget(spec, items)

    if args.dry_run:
        print(f"{spec.name}: {budget.render()}")
        print("\n  dry run: nothing was called and nothing was written.")
        return 0

    out = Path(args.out) if args.out else Path("results") / spec.name
    cache = ResponseCache(args.cache_dir, enabled=not args.no_cache)
    max_calls = args.max_calls if args.max_calls is not None else spec.budget.max_calls

    print(f"{spec.name}: {budget.render()}")
    outcome = run_experiment(
        spec,
        items,
        out,
        cache=cache,
        max_calls=max_calls,
        resume=not args.no_resume,
    )
    print(
        f"\n  wrote {outcome.trials_written:,} trial(s) to {outcome.results_dir} "
        f"({outcome.model_calls:,} model call(s), {outcome.cache_hits:,} cache hit(s), "
        f"{outcome.trials_skipped:,} already present)"
    )
    print(f"  determinism: {outcome.determinism} (observed, not assumed)")
    if outcome.errors:
        print(f"  {outcome.errors} trial(s) recorded a provider error", file=sys.stderr)

    if args.analyze:
        print()
        analyze_args = argparse.Namespace(
            results_dir=str(outcome.results_dir),
            model=None,
            control=None,
            alpha=spec.analysis.alpha,
            power=spec.analysis.power,
            resamples=spec.analysis.bootstrap_resamples,
            permutations=spec.analysis.permutations,
            seed=spec.trials.seed if spec.trials.seed is not None else 0,
            sesoi=spec.analysis.sesoi,
            simultaneous_ci=spec.analysis.simultaneous_ci,
            json=True,
            md=True,
        )
        return _cmd_analyze(analyze_args)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code; never raises for user error."""
    parser = build_parser()
    args = parser.parse_args(argv)

    handlers = {"power": _cmd_power, "analyze": _cmd_analyze, "run": _cmd_run}
    try:
        return handlers[args.command](args)
    except ErrorbarsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        # Out-of-range statistical arguments (a baseline of 1.5, an ICC of -0.2) surface
        # from the stats layer as ValueError. They are user error, not a defect, so they
        # get the same one-line treatment rather than a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\ninterrupted; anything already written is resumable", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
