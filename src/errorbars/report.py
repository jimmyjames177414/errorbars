"""Rendering: the terminal tables, ``report.json`` and ``report.md``.

Two rules govern everything here.

**Never round a confidence interval away.** ``-18.4pp`` on its own invites a reader to
treat it as precise. ``-18.4pp [-24.1, -12.9]`` does not. The interval is printed next
to every effect, in every format, with no flag to turn it off.

**The verdict column earns its place.** ``null`` and ``underpowered`` both look like
"not significant" in every other tool, and they mean opposite things: one is a finding,
the other is an admission. Each row carries the reason it got the verdict it did.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .ingest import IngestedRun
from .stats.effects import AnalysisResult, EffectEstimate
from .stats.power import PowerResult

__all__ = [
    "render_analysis_table",
    "render_power",
    "write_report_json",
    "write_report_md",
]

_VERDICT_LABEL = {
    "significant": "significant",
    "null": "null",
    "underpowered": "UNDERPOWERED",
}


def _pp(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:+.1f}"


def _pp_plain(value: float) -> str:
    if math.isnan(value):
        return "n/a"
    return f"{value * 100:.1f}"


def _p(value: float) -> str:
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def _column_widths(rows: list[list[str]], headers: list[str]) -> list[int]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    return widths


def _table(headers: list[str], rows: list[list[str]], aligns: str) -> str:
    widths = _column_widths(rows, headers)

    def line(cells: list[str]) -> str:
        parts = []
        for index, cell in enumerate(cells):
            width = widths[index]
            parts.append(cell.ljust(width) if aligns[index] == "l" else cell.rjust(width))
        return "  ".join(parts).rstrip()

    out = [line(headers), "  ".join("-" * width for width in widths)]
    out.extend(line(row) for row in rows)
    return "\n".join(out)


def render_analysis_table(run: IngestedRun, analysis: AnalysisResult) -> str:
    """The main ``errorbars analyze`` output."""
    headers = [
        "intervention",
        "effect",
        analysis.ci_label,
        "p (raw)",
        "p (Holm, family-wise)",
        "n",
        "verdict",
    ]
    rows: list[list[str]] = []
    for effect in analysis.effects:
        rows.append(
            [
                effect.intervention_id,
                f"{_pp(effect.effect)}pp",
                f"[{_pp(effect.ci_low)}, {_pp(effect.ci_high)}]",
                _p(effect.p_raw),
                _p(effect.p_adjusted),
                str(effect.n_items),
                _VERDICT_LABEL[effect.verdict],
            ]
        )

    control = analysis.effects[0] if analysis.effects else None
    lines = [
        f"{run.results_dir}",
        f"  tool          {run.tool_name} {run.tool_version} (CXS {run.cxs_version or 'unstated'})",
        f"  model         {run.model_key}",
        f"  scorer        {run.scorer}",
        f"  control       {analysis.control_id}"
        + (f"  ({control.control_rate:.1%} correct)" if control else ""),
        f"  design        {run.n_items} items x {run.repeats:g} repeats,"
        f" alpha={analysis.alpha:g}, power={analysis.power:.0%}",
    ]
    if run.skipped_unresolved:
        # Above the table on purpose. If a large share of a run never resolved, that has
        # to be known before any effect size below is read, not discovered in a footnote.
        lines.append(
            f"  UNRESOLVED    {run.skipped_inconclusive} inconclusive + "
            f"{run.skipped_error} error trial(s) skipped, excluded from every rate"
        )
    lines += [
        "",
        _table(headers, rows, "lrrrrrl"),
        "",
    ]

    for effect in analysis.effects:
        observed = (
            "n/a" if math.isnan(effect.mde_observed) else f"{_pp_plain(effect.mde_observed)}pp"
        )
        design = "n/a" if math.isnan(effect.mde_design) else f"{_pp_plain(effect.mde_design)}pp"
        lines.append(
            f"  {effect.intervention_id}: MDE {design} by design, {observed} achieved. "
            f"ICC {effect.icc:.2f}, pairing r {effect.pair_corr:.2f}"
        )
        lines.append(f"      {effect.reason}")

    if analysis.family_size > 1:
        lines.append("")
        if analysis.simultaneous_ci:
            lines += [
                f"  Intervals are simultaneous across {analysis.family_size} arms "
                f"(Bonferroni, each at",
                f"  alpha/{analysis.family_size} = {analysis.ci_alpha:.4g}), so they and "
                "the Holm p-values control the same",
                "  family-wise error rate.",
            ]
        else:
            lines += [
                "  The p-values are family-wise (Holm); the intervals are per-comparison. Each",
                "  interval holds on its own, so the chance at least one of them misses grows with",
                "  the number of arms. That is the usual default and is fine when you read one",
                "  interval at a time. Use --simultaneous-ci to put both columns on the "
                "same footing.",
            ]

    underpowered = [e for e in analysis.effects if e.verdict == "underpowered"]
    nulls = [e for e in analysis.effects if e.verdict == "null"]
    if nulls:
        lines += [
            "",
            "  null      = not significant AND the interval excludes an effect as large as "
            "the MDE.",
            f"              {', '.join(e.intervention_id for e in nulls)} did nothing, and this "
            "run was big enough to have seen it.",
        ]
    if underpowered:
        lines += [
            "",
            "  UNDERPOWERED = not significant, but the interval still admits an effect "
            "worth caring about.",
            f"              {', '.join(e.intervention_id for e in underpowered)} taught you "
            "nothing. Use `errorbars power` to size a run that would.",
        ]

    if analysis.notes:
        lines.append("")
        lines.append("  notes")
        for note in run.notes + analysis.notes:
            lines.append(f"    - {note}")

    return "\n".join(lines)


def render_power(
    result: PowerResult, *, sensitivity: list[tuple[float, float]] | None = None
) -> str:
    """The ``errorbars power`` output. Zero config, zero network, zero API key."""
    mde = "not detectable at any size" if math.isnan(result.mde) else f"{_pp_plain(result.mde)} pp"
    lines = [
        f"Power analysis  (alpha={result.alpha:g}, power={result.power:.0%}, two-sided, paired)",
        "",
        f"  design              {result.n_items} items x {result.repeats} repeats "
        f"= {result.n_items * result.repeats:,} observations per arm",
        f"  baseline            {result.baseline:.1%}",
        f"  ICC (assumed)       {result.icc:.2f}   correlation of repeats within an item",
        f"  pairing r (assumed) {result.pair_corr:.2f}   correlation of item difficulty "
        "across arms",
        f"  design effect       {result.design_effect:.2f}   inflation from repeated measurement",
        f"  effective n         {result.effective_n:,.0f}   independent-equivalent "
        "observations per arm",
        "",
        f"  minimum detectable effect   {mde}",
    ]
    if result.target_effect is not None and result.items_for_target is not None:
        lines.append(
            f"  to detect {result.target_effect * 100:.1f} pp you need    "
            f"~{result.items_for_target:,} items (at {result.repeats} repeats)"
        )

    if not math.isnan(result.mde):
        lines += [
            "",
            f"  A {result.mde * 100:.1f} pp result at this N is at the edge of what the run "
            "can see.",
            f"  Anything under {result.mde * 100 / 2:.1f} pp is noise you cannot "
            "distinguish from zero.",
        ]

    if sensitivity:
        lines += ["", "  sensitivity to the pairing assumption"]
        for pair_corr, sensitivity_mde in sensitivity:
            rendered = "n/a" if math.isnan(sensitivity_mde) else f"{sensitivity_mde * 100:5.1f} pp"
            lines.append(f"    pairing r = {pair_corr:.1f}   MDE {rendered}")
        lines.append(
            "    Higher pairing correlation means a tighter MDE. The default of 0.00 assumes"
        )
        lines.append(
            "    pairing buys nothing, so the headline number is never optimistic. Run `errorbars"
        )
        lines.append("    analyze` on a pilot to measure the real value.")

    return "\n".join(lines)


def _effect_dict(effect: EffectEstimate) -> dict[str, Any]:
    return {
        "intervention_id": effect.intervention_id,
        "n_items": effect.n_items,
        "repeats": effect.repeats,
        "control_rate": effect.control_rate,
        "treatment_rate": effect.treatment_rate,
        "control_rate_ci": [effect.control_ci.low, effect.control_ci.high],
        "treatment_rate_ci": [effect.treatment_ci.low, effect.treatment_ci.high],
        "rate_ci_method": effect.rate_ci_method,
        "effect": effect.effect,
        "ci_low": effect.ci_low,
        "ci_high": effect.ci_high,
        "p_raw": effect.p_raw,
        "p_holm": effect.p_adjusted,
        "icc": effect.icc,
        "pair_corr": effect.pair_corr,
        "mde_observed": None if math.isnan(effect.mde_observed) else effect.mde_observed,
        "mde_design": None if math.isnan(effect.mde_design) else effect.mde_design,
        "sesoi": None if math.isnan(effect.sesoi) else effect.sesoi,
        "verdict": effect.verdict,
        "reason": effect.reason,
    }


def write_report_json(path: Path, run: IngestedRun, analysis: AnalysisResult) -> Path:
    """Write ``report.json``: everything the table shows, plus what it does not fit."""
    payload = {
        "cxs_version": run.cxs_version,
        "experiment_id": run.manifest.get("experiment_id"),
        "analysed_by": {"name": "errorbars", "version": __import__("errorbars").__version__},
        "source_tool": {"name": run.tool_name, "version": run.tool_version},
        "model": run.model_key,
        "scorer": run.scorer,
        "control_intervention_id": analysis.control_id,
        "n_items": run.n_items,
        "repeats": run.repeats,
        "skipped_inconclusive": run.skipped_inconclusive,
        "skipped_error": run.skipped_error,
        "statistics": {
            "alpha": analysis.alpha,
            "power": analysis.power,
            "ci_method": "percentile bootstrap over items, paired",
            "ci_scope": "simultaneous" if analysis.simultaneous_ci else "per-comparison",
            "ci_alpha": analysis.ci_alpha,
            "bootstrap_resamples": analysis.bootstrap_resamples,
            "test": "paired permutation (sign-flip)",
            "permutations": analysis.permutations,
            "multiplicity": "holm",
            "family_size": analysis.family_size,
        },
        "effects": [_effect_dict(effect) for effect in analysis.effects],
        "notes": run.notes + analysis.notes,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_report_md(path: Path, run: IngestedRun, analysis: AnalysisResult) -> Path:
    """Write ``report.md``: the same content, pasteable into a PR or an issue."""
    header = (
        f"| intervention | effect | {analysis.ci_label} | p (raw) | "
        "p (Holm, family-wise) | n items | MDE | verdict |"
    )
    separator = "|---|---:|---:|---:|---:|---:|---:|---|"
    rows = []
    for effect in analysis.effects:
        mde = "n/a" if math.isnan(effect.mde_design) else f"{_pp_plain(effect.mde_design)}pp"
        rows.append(
            f"| `{effect.intervention_id}` | {_pp(effect.effect)}pp | "
            f"[{_pp(effect.ci_low)}, {_pp(effect.ci_high)}] | {_p(effect.p_raw)} | "
            f"{_p(effect.p_adjusted)} | {effect.n_items} | {mde} | "
            f"**{_VERDICT_LABEL[effect.verdict]}** |"
        )

    control_rate = analysis.effects[0].control_rate if analysis.effects else float("nan")
    lines = [
        f"# {run.results_dir.name}",
        "",
        f"- source tool: `{run.tool_name}` {run.tool_version} "
        f"(CXS {run.cxs_version or 'unstated'})",
        f"- model: `{run.model_key}`",
        f"- scorer: `{run.scorer}`",
        f"- control arm: `{analysis.control_id}`, {control_rate:.1%} correct",
        f"- design: {run.n_items} items x {run.repeats:g} repeats, "
        f"alpha = {analysis.alpha:g}, target power = {analysis.power:.0%}",
        *(
            [
                f"- **unresolved: {run.skipped_inconclusive} inconclusive + "
                f"{run.skipped_error} error trial(s) skipped**, excluded from every rate "
                "and denominator"
            ]
            if run.skipped_unresolved
            else []
        ),
        f"- intervals: percentile bootstrap over items, "
        f"{analysis.bootstrap_resamples:,} resamples, paired, "
        + (
            f"**simultaneous** across {analysis.family_size} arm(s) at "
            f"alpha/{analysis.family_size} = {analysis.ci_alpha:.4g} (Bonferroni)"
            if analysis.simultaneous_ci
            else f"**per-comparison** at alpha = {analysis.alpha:g}"
        ),
        f"- test: paired permutation, {analysis.permutations:,} sign-flips, Holm-adjusted across "
        f"{len(analysis.effects)} intervention(s)",
        "",
        header,
        separator,
        *rows,
        "",
        "## Two different error rates in one table",
        "",
        (
            "The p-value column is **family-wise**: Holm controls the chance of *any* false "
            "positive across all interventions tested. The interval column is "
            + (
                "**simultaneous**, so it is on the same footing -- all of the intervals hold "
                "together at the stated confidence."
                if analysis.simultaneous_ci
                else "**per-comparison**, so it is not on the same footing -- each interval "
                "holds on its own, and the chance that at least one of them misses grows "
                "with the number of arms. Re-run with `--simultaneous-ci` if you are "
                "reading the intervals as a screen across arms rather than one at a time."
            )
        ),
        "",
        "## How to read the verdict",
        "",
        "- **significant** -- the Holm-adjusted p-value clears alpha.",
        "- **null** -- not significant, *and* the interval excludes an effect as large as the "
        "MDE. The intervention did nothing and this run was big enough to have noticed.",
        "- **UNDERPOWERED** -- not significant, but the interval still admits an effect worth "
        "caring about. Nothing was learned; size the next run with `errorbars power`.",
        "",
    ]
    if run.notes or analysis.notes:
        lines.append("## Notes")
        lines.append("")
        lines.extend(f"- {note}" for note in run.notes + analysis.notes)
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
