from __future__ import annotations

import json
from pathlib import Path

from synthetic_checkpoints import (
    bad_values_checkpoint,
    before_after_checkpoints,
    corrupt_checkpoint,
    normal_checkpoint,
)
from typer.testing import CliRunner

from ckptguard.cli import app

runner = CliRunner()


def test_stats_json_stdout_with_synthetic_normal_checkpoint(tmp_path: Path):
    checkpoint = normal_checkpoint(tmp_path)

    result = runner.invoke(app, ["stats", str(checkpoint), "--json", "--no-cache"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == "1.1"
    assert payload["summary"]["tensor_count"] == 4
    assert {tensor["name"] for tensor in payload["tensors"]} >= {
        "embed.weight",
        "layer.lora_A.weight",
        "layer.lora_B.weight",
    }


def test_audit_json_stdout_with_synthetic_bad_values(tmp_path: Path):
    checkpoint = bad_values_checkpoint(tmp_path)

    result = runner.invoke(
        app,
        ["audit", str(checkpoint), "--fail-on", "nan,inf", "--json", "--no-cache"],
    )

    assert result.exit_code == 1
    payload = json.loads(result.stdout)
    categories = {finding["category"] for finding in payload["findings"]}
    assert {"nan", "inf", "suspicious-values", "lora-all-zero"} <= categories
    assert not payload["passed"]


def test_diff_json_stdout_with_synthetic_before_after(tmp_path: Path):
    before, after = before_after_checkpoints(tmp_path)

    result = runner.invoke(app, ["diff", str(before), str(after), "--json", "--no-cache"])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    by_name = {tensor["name"]: tensor for tensor in payload["tensors"]}
    assert "shape" in by_name["shape.changed"]["changes"]
    assert by_name["dtype.changed"]["changes"] == ["dtype", "values"]
    assert by_name["added_removed.added"]["status"] == "added"
    assert by_name["added_removed.removed"]["status"] == "removed"
    assert by_name["norm.spike"]["norm_delta"] > 10


def test_report_top_limits_rendered_diff_rows(tmp_path: Path):
    before, after = before_after_checkpoints(tmp_path)
    output = tmp_path / "report.html"

    result = runner.invoke(
        app,
        [
            "report",
            str(before),
            str(after),
            "--html",
            "--top",
            "3",
            "--output",
            str(output),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    html = output.read_text(encoding="utf-8")
    assert "shape.changed" in html
    assert "added_removed.added" in html
    assert "added_removed.removed" in html
    assert "value.drift" not in html


def test_ci_synthetic_bad_values_fails_and_normal_passes(tmp_path: Path):
    normal = normal_checkpoint(tmp_path)
    bad = bad_values_checkpoint(tmp_path)

    normal_result = runner.invoke(app, ["ci", str(normal), "--json", "--no-cache"])
    bad_result = runner.invoke(
        app,
        ["ci", str(bad), "--fail-on", "nan,inf", "--json", "--no-cache"],
    )

    assert normal_result.exit_code == 0
    assert json.loads(normal_result.stdout)["passed"]
    assert bad_result.exit_code == 1
    assert not json.loads(bad_result.stdout)["passed"]


def test_unknown_fail_on_category_exits_with_usage_error(tmp_path: Path):
    checkpoint = normal_checkpoint(tmp_path)

    result = runner.invoke(
        app,
        ["audit", str(checkpoint), "--fail-on", "unknown-category", "--no-cache"],
    )

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Unknown --fail-on category" in result.stderr


def test_corrupt_safetensors_exits_with_read_error(tmp_path: Path):
    checkpoint = corrupt_checkpoint(tmp_path)

    result = runner.invoke(app, ["stats", str(checkpoint), "--no-cache"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Could not read tensors" in result.stderr
    assert str(checkpoint) in result.stderr


def test_non_safetensors_path_exits_with_supported_format_error(tmp_path: Path):
    path = tmp_path / "model.bin"
    path.write_bytes(b"not a checkpoint")

    result = runner.invoke(app, ["stats", str(path), "--no-cache"])

    assert result.exit_code == 2
    assert result.stdout == ""
    assert "Only .safetensors files are supported" in result.stderr
