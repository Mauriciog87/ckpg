from __future__ import annotations

import ml_dtypes
import numpy as np
import pytest

from ckptguard.errors import CkptGuardError
from ckptguard.stats.diff_stats import diff_checkpoints
from ckptguard.stats.tensor_stats import build_stats_report


def test_bfloat16_stats_include_numeric_and_nonfinite_metrics(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "bf16.safetensors",
        {
            "finite": np.array([0.0, 1.0, 2.0], dtype=ml_dtypes.bfloat16),
            "nonfinite": np.array([np.nan, np.inf], dtype=ml_dtypes.bfloat16),
        },
    )

    report = build_stats_report(checkpoint, use_cache=False)
    by_name = {tensor.name: tensor for tensor in report.tensors}

    assert by_name["finite"].dtype == "bfloat16"
    assert by_name["finite"].mean == pytest.approx(1.0)
    assert by_name["finite"].l2_norm == pytest.approx(np.sqrt(5.0))
    assert by_name["nonfinite"].nan_count == 1
    assert by_name["nonfinite"].inf_count == 1


def test_bfloat16_diff_calculates_numeric_deltas(checkpoint_factory):
    before = checkpoint_factory(
        "before-bf16.safetensors",
        {"x": np.array([1.0, 2.0], dtype=ml_dtypes.bfloat16)},
    )
    after = checkpoint_factory(
        "after-bf16.safetensors",
        {"x": np.array([1.0, 3.0], dtype=ml_dtypes.bfloat16)},
    )

    tensor = diff_checkpoints(before, after, use_cache=False).tensors[0]

    assert tensor.status == "changed"
    assert tensor.norm_delta == pytest.approx(np.sqrt(10.0) - np.sqrt(5.0))
    assert tensor.cosine_distance is not None


def test_float8_returns_explicit_unsupported_dtype_error(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "fp8.safetensors",
        {"x": np.array([0.0, 1.0], dtype=ml_dtypes.float8_e4m3fn)},
    )

    with pytest.raises(CkptGuardError, match=r"Unsupported safetensors dtype.*FP8"):
        build_stats_report(checkpoint, use_cache=False)
