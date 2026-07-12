from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from ckptguard.cli import app

runner = CliRunner()


def test_stats_rejects_checkpoint_as_json_output(checkpoint_factory):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    original = checkpoint.read_bytes()

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--json-output", str(checkpoint), "--no-cache"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Output path conflicts with a protected input" in result.stderr
    assert checkpoint.read_bytes() == original


def test_stats_rejects_existing_alias_of_checkpoint(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    alias = tmp_path / "alias.json"
    try:
        os.link(checkpoint, alias)
    except OSError:
        pytest.skip("Hard links are unavailable on this filesystem")

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--json-output", str(alias), "--no-cache"],
    )

    assert result.exit_code == 2
    assert "Output path conflicts with a protected input" in result.stderr


def test_report_rejects_colliding_html_and_json_outputs(checkpoint_factory, tmp_path: Path):
    before = checkpoint_factory("before.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    after = checkpoint_factory("after.safetensors", {"x": np.array([2.0], dtype=np.float32)})
    output = tmp_path / "report.out"

    result = runner.invoke(
        app,
        [
            "report",
            str(before),
            str(after),
            "--html",
            "--output",
            str(output),
            "--json-output",
            str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 2
    assert not output.exists()


def test_json_output_replaces_existing_file_atomically(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    output = tmp_path / "stats.json"
    output.write_text("old", encoding="utf-8")

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--json-output", str(output), "--no-cache"],
    )

    assert result.exit_code == 0
    assert output.read_text(encoding="utf-8").startswith("{")
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_output_directory_error_is_clean(checkpoint_factory, tmp_path: Path):
    checkpoint = checkpoint_factory("model.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    parent = tmp_path / "not-a-directory"
    parent.write_text("file", encoding="utf-8")
    output = parent / "stats.json"

    result = runner.invoke(
        app,
        ["stats", str(checkpoint), "--json-output", str(output), "--no-cache"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Could not write output file" in result.stderr
