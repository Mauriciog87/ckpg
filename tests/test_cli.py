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
