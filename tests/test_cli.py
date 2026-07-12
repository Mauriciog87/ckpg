from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from ckptguard.cli import app

runner = CliRunner()


def test_stats_command_writes_json(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    output = tmp_path / "stats.json"

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--json-output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["tensor_count"] == 1


def test_diff_command_writes_json(checkpoint_factory, tmp_path: Path):
    before = checkpoint_factory("before.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    after = checkpoint_factory("after.safetensors", {"x": np.array([2.0], dtype=np.float32)})
    output = tmp_path / "diff.json"

    result = runner.invoke(
        app,
        ["diff", str(before), str(after), "--json-output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["changed"] == 1


def test_audit_command_returns_failure_for_selected_findings(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("bad.safetensors", {"x": np.array([np.nan], dtype=np.float32)})
    output = tmp_path / "audit.json"

    result = runner.invoke(
        app,
        ["audit", str(checkpoint), "--fail-on", "nan", "--json-output", str(output), "--no-cache"],
    )

    assert result.exit_code == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert not payload["passed"]


def test_ci_command_returns_failure_for_selected_findings(checkpoint_factory):
    checkpoint = checkpoint_factory("bad.safetensors", {"x": np.array([np.inf], dtype=np.float32)})

    result = runner.invoke(app, ["ci", str(checkpoint), "--fail-on", "inf", "--no-cache"])

    assert result.exit_code == 1


def test_ci_command_prints_success_message(checkpoint_factory):
    checkpoint = checkpoint_factory("good.safetensors", {"x": np.array([1.0], dtype=np.float32)})

    result = runner.invoke(app, ["ci", str(checkpoint), "--no-cache"])

    assert result.exit_code == 0
    assert "ckpg CI checks passed" in result.stdout


def test_report_command_writes_html(checkpoint_factory, tmp_path: Path):
    before = checkpoint_factory("before.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    after = checkpoint_factory("after.safetensors", {"x": np.array([2.0], dtype=np.float32)})
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", str(before), str(after), "--html", "--output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    html = output.read_text(encoding="utf-8")
    assert "ckpg report" in html
    assert "x" in html
    assert str(tmp_path) not in html
    assert "before.safetensors" in html
    assert "after.safetensors" in html


def test_report_combines_absolute_and_differential_findings(checkpoint_factory, tmp_path: Path):
    before = checkpoint_factory(
        "before.safetensors",
        {"shape": np.zeros((2,), dtype=np.float32)},
    )
    after = checkpoint_factory(
        "after.safetensors",
        {
            "shape": np.zeros((3,), dtype=np.float32),
            "bad.suspicious": np.array([1_500_000.0], dtype=np.float32),
        },
    )
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        ["report", str(before), str(after), "--html", "--output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    html = output.read_text(encoding="utf-8")
    assert "shape-drift" in html
    assert "suspicious-values" in html


def test_ci_with_baseline_fails_on_structural_drift(checkpoint_factory):
    baseline = checkpoint_factory(
        "baseline.safetensors",
        {
            "changed": np.zeros((2, 2), dtype=np.float32),
            "removed": np.ones((1,), dtype=np.float32),
        },
    )
    candidate = checkpoint_factory(
        "candidate.safetensors",
        {
            "changed": np.zeros((4,), dtype=np.float32),
            "added": np.ones((1,), dtype=np.float32),
        },
    )

    result = runner.invoke(
        app,
        ["ci", str(candidate), "--baseline", str(baseline), "--json", "--no-cache"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.1"
    assert payload["baseline_file"]["path"] == str(baseline.resolve())
    shape_tensors = {
        finding["tensor"] for finding in payload["findings"] if finding["category"] == "shape-drift"
    }
    assert shape_tensors == {"added", "changed", "removed"}


def test_diff_only_fail_category_requires_baseline(checkpoint_factory):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.ones((1,), dtype=np.float32)})

    result = runner.invoke(
        app,
        ["ci", str(checkpoint), "--fail-on", "shape-drift", "--no-cache"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "requires --baseline" in result.stderr


def test_audit_with_baseline_combines_candidate_and_diff_findings(checkpoint_factory):
    baseline = checkpoint_factory(
        "baseline.safetensors",
        {"shape": np.zeros((2,), dtype=np.float32)},
    )
    candidate = checkpoint_factory(
        "candidate.safetensors",
        {
            "shape": np.zeros((3,), dtype=np.float32),
            "bad.suspicious": np.array([1_500_000.0], dtype=np.float32),
        },
    )

    result = runner.invoke(
        app,
        ["audit", str(candidate), "--baseline", str(baseline), "--json", "--no-cache"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    categories = {finding["category"] for finding in payload["findings"]}
    assert {"shape-drift", "suspicious-values"} <= categories


def test_corrupt_cache_error_is_written_to_stderr(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.ones((1,), dtype=np.float32)})
    cache = tmp_path / "cache.sqlite"
    cache.write_bytes(b"not a sqlite database")

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--cache-db", str(cache)],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Cache database is invalid or unavailable" in result.stderr
