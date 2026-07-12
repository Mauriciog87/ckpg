from __future__ import annotations

import numpy as np
import pytest

from ckptguard.readers.safetensors_reader import SafeTensorsFile
from ckptguard.stats.tensor_stats import build_stats_report


def test_reader_lists_tensor_keys(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "keys.safetensors",
        {
            "b": np.array([1.0], dtype=np.float32),
            "a": np.array([2.0], dtype=np.float32),
        },
    )

    assert SafeTensorsFile(checkpoint).keys() == ["a", "b"]


def test_stats_report_contains_expected_tensor_metrics(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "model.safetensors",
        {
            "a": np.array([[1.0, 2.0], [0.0, 4.0]], dtype=np.float32),
            "b": np.array([0, 0, 0], dtype=np.int64),
        },
    )

    report = build_stats_report(checkpoint, use_cache=False)
    by_name = {tensor.name: tensor for tensor in report.tensors}

    assert report.summary.tensor_count == 2
    assert by_name["a"].shape == [2, 2]
    assert by_name["a"].dtype == "float32"
    assert by_name["a"].numel == 4
    assert by_name["a"].min == 0.0
    assert by_name["a"].max == 4.0
    assert by_name["a"].nan_count == 0
    assert by_name["a"].inf_count == 0
    assert by_name["b"].zero_ratio == 1.0


def test_stats_report_uses_json_safe_nulls_for_nonfinite_values(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "bad.safetensors",
        {"bad": np.array([1.0, np.nan, np.inf], dtype=np.float32)},
    )

    report = build_stats_report(checkpoint, use_cache=False)
    tensor = report.tensors[0]
    json_payload = report.model_dump_json()

    assert tensor.nan_count == 1
    assert tensor.inf_count == 1
    assert tensor.mean is None
    assert "NaN" not in json_payload
    assert "Infinity" not in json_payload


def test_chunked_stats_match_numpy_for_large_tensor(checkpoint_factory):
    values = np.linspace(-100.0, 250.0, 1_048_583, dtype=np.float32)
    checkpoint = checkpoint_factory("large.safetensors", {"large": values})

    tensor = build_stats_report(checkpoint, use_cache=False).tensors[0]
    reference = values.astype(np.float64)

    assert tensor.min == pytest.approx(float(reference.min()))
    assert tensor.max == pytest.approx(float(reference.max()))
    assert tensor.mean == pytest.approx(float(reference.mean()), rel=1e-12)
    assert tensor.std == pytest.approx(float(reference.std()), rel=1e-12)
    assert tensor.l2_norm == pytest.approx(float(np.linalg.norm(reference)), rel=1e-12)
