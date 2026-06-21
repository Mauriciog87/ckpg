from __future__ import annotations

from pathlib import Path

import numpy as np
from safetensors.numpy import save_file


def normal_checkpoint(directory: Path) -> Path:
    path = directory / "normal.safetensors"
    save_file(
        {
            "embed.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
            "layer.lora_A.weight": np.ones((2, 4), dtype=np.float32),
            "layer.lora_B.weight": np.ones((4, 2), dtype=np.float32),
            "zeros.bias": np.zeros((4,), dtype=np.float32),
        },
        path,
    )
    return path


def bad_values_checkpoint(directory: Path) -> Path:
    path = directory / "bad_values.safetensors"
    save_file(
        {
            "bad.nan_inf": np.array([0.0, np.nan, np.inf], dtype=np.float32),
            "bad.suspicious": np.array([1_500_000.0, 2.0], dtype=np.float32),
            "layer.lora_A.weight": np.zeros((2, 4), dtype=np.float32),
            "layer.lora_B.weight": np.ones((4, 3), dtype=np.float16),
        },
        path,
    )
    return path


def before_after_checkpoints(directory: Path) -> tuple[Path, Path]:
    before = directory / "before.safetensors"
    after = directory / "after.safetensors"
    save_file(
        {
            "added_removed.removed": np.array([1.0], dtype=np.float32),
            "dtype.changed": np.array([1.0, 2.0], dtype=np.float32),
            "norm.spike": np.array([1.0, 0.0], dtype=np.float32),
            "shape.changed": np.zeros((2, 2), dtype=np.float32),
            "value.drift": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        },
        before,
    )
    save_file(
        {
            "added_removed.added": np.array([1.0], dtype=np.float32),
            "dtype.changed": np.array([1, 2], dtype=np.int64),
            "norm.spike": np.array([20.0, 0.0], dtype=np.float32),
            "shape.changed": np.zeros((4,), dtype=np.float32),
            "value.drift": np.array([1.0, 2.0, 4.0], dtype=np.float32),
        },
        after,
    )
    return before, after


def corrupt_checkpoint(directory: Path) -> Path:
    path = directory / "corrupt.safetensors"
    path.write_bytes(b"not a valid safetensors file")
    return path
