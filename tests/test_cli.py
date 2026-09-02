"""The command line, with ``errorbars power`` held to its promise.

The flagship claim is "zero config, zero network, zero API key". That is only worth
printing in a README if something checks it, so the first test below runs the command
with sockets removed from the process and the home directory pointed somewhere empty.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from errorbars.cli import main

FOREIGN = Path(__file__).parent / "fixtures" / "foreign_tool_results"

EXAMPLE_SPEC = """
version: "0.1"
name: cli-demo
dataset: { path: data.jsonl }
models:
  - { provider: mock, model: sim }
interventions:
  - { id: control, kind: noop }
  - { id: strip-negation, kind: remove,
      selector: { type: lexical_class, value: negation } }
scorer: first_word
trials: { repeats: 2, seed: 3 }
"""


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    rows = [
        json.dumps(
            {
                "id": f"item-{index}",
                "prompt": f"Is {index + 5} not greater than {index + 30}? "
                "Answer with one word: true or false.",
                "expected": "true",
            }
        )
        for index in range(12)
    ]
    (tmp_path / "data.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    (tmp_path / "experiment.yaml").write_text(EXAMPLE_SPEC, encoding="utf-8")
    return tmp_path


# --------------------------------------------------------------------------------------
# power
# --------------------------------------------------------------------------------------


def test_power_needs_no_network_no_key_and_no_config(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The headline promise, enforced: sockets removed, environment emptied."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("errorbars power attempted a network call")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    assert main(["power", "--items", "200", "--repeats", "5", "--baseline", "0.80"]) == 0

    out = capsys.readouterr().out
    assert "minimum detectable effect" in out
    assert "effective n" in out
    assert "design effect" in out
    assert list(tmp_path.iterdir()) == [], "power must not write anything"


def test_power_defaults_produce_a_usable_answer(capsys: pytest.CaptureFixture[str]) -> None:
    """`errorbars power` with no arguments at all has to say something sensible."""
    assert main(["power"]) == 0
    out = capsys.readouterr().out
    assert "200 items x 1 repeats" in out
    assert "pp" in out


def test_power_reports_required_items_for_a_target_effect(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["power", "--items", "200", "--baseline", "0.80", "--detect", "1pp"]) == 0
    out = capsys.readouterr().out
    assert "to detect 1.0 pp you need" in out


@pytest.mark.parametrize("spelling", ["0.02", "2%", "2pp"])
def test_effect_sizes_accept_all_three_spellings(
    capsys: pytest.CaptureFixture[str], spelling: str
) -> None:
    """Getting this wrong by 100x silently is the mistake the whole tool exists to stop."""
    assert main(["power", "--items", "400", "--baseline", "0.80", "--detect", spelling]) == 0
    assert "to detect 2.0 pp you need" in capsys.readouterr().out


def test_power_shows_the_pairing_sensitivity_block(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "--items", "200", "--repeats", "5"]) == 0
    out = capsys.readouterr().out
    assert "sensitivity to the pairing assumption" in out
    assert "pairing r = 0.9" in out


def test_sensitivity_block_can_be_suppressed(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["power", "--items", "200", "--no-sensitivity"]) == 0
    assert "sensitivity to the pairing assumption" not in capsys.readouterr().out


def test_power_labels_its_assumptions_as_assumptions(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """ICC and pairing are guesses until measured; the output must not pretend otherwise."""
    assert main(["power", "--items", "200", "--repeats", "5"]) == 0
    out = capsys.readouterr().out
    assert "ICC (assumed)" in out
    assert "pairing r (assumed)" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["power", "--baseline", "1.5"],
        ["power", "--icc", "-0.2"],
        ["power", "--pair-corr", "2.0"],
        ["power", "--items", "0"],
        ["power", "--alpha", "1.0"],
    ],
)
def test_bad_power_arguments_exit_non_zero_without_a_traceback(
    capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------------------


def test_analyze_reads_the_foreign_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["analyze", str(FOREIGN), "--resamples", "1000", "--permutations", "1000"]) == 0
    out = capsys.readouterr().out
    assert "contextprobe" in out
    assert "shuffle-context" in out
    assert "95% CI" in out
    assert "verdict" in out


def test_analyze_writes_report_files_on_request(tmp_path: Path) -> None:
    import shutil

    destination = tmp_path / "run"
    shutil.copytree(FOREIGN, destination)
    assert (
        main(
            [
                "analyze",
                str(destination),
                "--json",
                "--md",
                "--resamples",
                "500",
                "--permutations",
                "500",
            ]
        )
        == 0
    )

    report = json.loads((destination / "report.json").read_text(encoding="utf-8"))
    assert report["source_tool"]["name"] == "contextprobe"
    assert report["statistics"]["ci_method"].startswith("percentile bootstrap")
    assert {effect["verdict"] for effect in report["effects"]} <= {
        "significant",
        "null",
        "underpowered",
    }

    markdown = (destination / "report.md").read_text(encoding="utf-8")
    assert "| intervention |" in markdown
    assert "UNDERPOWERED" in markdown or "null" in markdown


def test_analyze_never_prints_an_effect_without_its_interval(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["analyze", str(FOREIGN), "--resamples", "500", "--permutations", "500"]) == 0
    out = capsys.readouterr().out
    for line in out.splitlines():
        if "pp  " in line and "[" not in line and "MDE" not in line:
            pytest.fail(f"an effect was printed without its interval: {line!r}")


def test_analyze_on_a_missing_directory_exits_one_with_a_message(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    assert main(["analyze", str(tmp_path / "nope")]) == 1
    assert "error:" in capsys.readouterr().err


# --------------------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------------------


def test_dry_run_reports_the_exact_call_count_and_writes_nothing(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    before = set(workspace.iterdir())
    assert main(["run", "experiment.yaml", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "48 (exact)" in out, "2 arms x 12 items x 2 repeats"
    assert "estimated" in out, "token counts are estimates and must say so"
    assert "nothing was called and nothing was written" in out
    assert set(workspace.iterdir()) == before


def test_run_then_analyze_end_to_end(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("XDG_CACHE_HOME", str(workspace / "cache"))
    assert main(["run", "experiment.yaml", "--out", "out", "--analyze"]) == 0

    out = capsys.readouterr().out
    assert "determinism: stochastic (observed, not assumed)" in out
    assert "strip-negation" in out
    assert (workspace / "out" / "manifest.json").is_file()
    assert (workspace / "out" / "report.json").is_file()


def test_max_calls_stops_the_run_and_says_how_to_resume(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    assert main(["run", "experiment.yaml", "--out", "out", "--max-calls", "10", "--no-cache"]) == 1
    assert "Re-run the same command to resume" in capsys.readouterr().err


def test_a_spec_without_a_control_arm_warns_on_stderr(
    workspace: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    (workspace / "experiment.yaml").write_text(
        EXAMPLE_SPEC.replace("  - { id: control, kind: noop }\n", ""), encoding="utf-8"
    )
    monkeypatch.chdir(workspace)
    assert main(["run", "experiment.yaml", "--dry-run"]) == 0
    assert "no control arm" in capsys.readouterr().err


def test_version_flag_reports_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    from errorbars import __version__

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_subcommand_is_an_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])
