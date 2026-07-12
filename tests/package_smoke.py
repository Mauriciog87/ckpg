from __future__ import annotations

import importlib.resources
import json
import subprocess
import tempfile
from importlib.metadata import distribution
from pathlib import Path

import ml_dtypes
import numpy as np
from safetensors.numpy import save_file


def run_command(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, check=True, capture_output=True, text=True)


def write_checkpoint(path: Path, value: float) -> None:
    save_file(
        {
            "layer.weight": np.array([value, value + 1.0], dtype=np.float32),
            "layer.lora_A.weight": np.ones((2, 4), dtype=np.float32),
            "layer.lora_B.weight": np.ones((4, 2), dtype=np.float32),
            "layer.bf16.weight": np.array([value, value + 1.0], dtype=ml_dtypes.bfloat16),
        },
        path,
    )


def console_scripts() -> dict[str, str]:
    return {
        entry.name: entry.value
        for entry in distribution("ckptguard").entry_points
        if entry.group == "console_scripts"
    }


def main() -> None:
    import ckptguard

    scripts = console_scripts()
    if scripts.get("ckpg") != "ckptguard.cli:main":
        raise AssertionError("ckpg console script is missing from installed package metadata")
    if "ckptguard" in scripts:
        raise AssertionError("ckptguard console script should not be declared")

    template = importlib.resources.files("ckptguard").joinpath("reports/templates/report.html.j2")
    if not template.is_file():
        raise AssertionError("HTML report template is missing from installed package")

    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        before = directory / "before.safetensors"
        after = directory / "after.safetensors"
        report = directory / "report.html"
        write_checkpoint(before, 1.0)
        write_checkpoint(after, 2.0)

        help_result = run_command(["ckpg", "--help"])
        if "stats" not in help_result.stdout:
            raise AssertionError("ckpg help output does not list stats command")

        stats_result = run_command(["ckpg", "stats", str(before), "--json", "--no-cache"])
        stats_payload = json.loads(stats_result.stdout)
        if stats_payload["summary"]["tensor_count"] != 4:
            raise AssertionError("unexpected tensor count in stats JSON")
        bf16_stats = next(
            tensor for tensor in stats_payload["tensors"] if tensor["name"] == "layer.bf16.weight"
        )
        if bf16_stats["dtype"] != "bfloat16" or bf16_stats["mean"] is None:
            raise AssertionError("BF16 statistics are unavailable")

        run_command(
            [
                "ckpg",
                "report",
                str(before),
                str(after),
                "--html",
                "--output",
                str(report),
                "--no-cache",
            ]
        )
        if "ckpg report" not in report.read_text(encoding="utf-8"):
            raise AssertionError("HTML report was not generated correctly")

    if not ckptguard.__version__:
        raise AssertionError("ckptguard package version is empty")


if __name__ == "__main__":
    main()
