from __future__ import annotations

import numpy as np
import pytest

from ckptguard.policies.audit_policy import (
    audit_combined_report,
    audit_diff_report,
    audit_stats_report,
)
from ckptguard.readers.safetensors_reader import SafeTensorsFile
from ckptguard.stats.diff_stats import _cosine_distance, diff_checkpoints
from ckptguard.stats.tensor_stats import build_stats_report


def test_diff_detects_shape_dtype_and_value_changes(checkpoint_factory):
    before = checkpoint_factory(
        "before.safetensors",
        {
            "same": np.array([1.0, 2.0], dtype=np.float32),
            "shape": np.zeros((2, 2), dtype=np.float32),
            "removed": np.array([1], dtype=np.int64),
        },
    )
    after = checkpoint_factory(
        "after.safetensors",
        {
            "same": np.array([1.0, 3.0], dtype=np.float32),
            "shape": np.zeros((4,), dtype=np.float32),
            "added": np.array([1], dtype=np.int64),
        },
    )

    report = diff_checkpoints(before, after, use_cache=False)
    by_name = {diff.name: diff for diff in report.tensors}

    assert by_name["same"].status == "changed"
    assert "values" in by_name["same"].changes
    assert "shape" in by_name["shape"].changes
    assert by_name["added"].status == "added"
    assert by_name["removed"].status == "removed"


def test_audit_flags_nan_inf_and_lora_anomalies(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "adapter.safetensors",
        {
            "layer.lora_A.weight": np.zeros((2, 3), dtype=np.float32),
            "layer.lora_B.weight": np.ones((4, 3), dtype=np.float16),
            "bad": np.array([np.nan, np.inf], dtype=np.float32),
        },
    )

    stats_report = build_stats_report(checkpoint, use_cache=False)
    report = audit_stats_report(stats_report, fail_on=["nan", "inf"])
    categories = {finding.category for finding in report.findings}

    assert not report.passed
    assert {
        "nan",
        "inf",
        "lora-rank-mismatch",
        "lora-dtype-mismatch",
        "lora-all-zero",
    } <= categories


def test_diff_audit_fails_shape_drift(checkpoint_factory):
    before = checkpoint_factory("before.safetensors", {"x": np.zeros((2, 2), dtype=np.float32)})
    after = checkpoint_factory("after.safetensors", {"x": np.zeros((4,), dtype=np.float32)})

    diff_report = diff_checkpoints(before, after, use_cache=False)
    audit_report = audit_diff_report(diff_report, fail_on=["shape-drift"])

    assert not audit_report.passed
    assert audit_report.findings[0].category == "shape-drift"


def test_diff_audit_uses_rms_for_norm_spikes(checkpoint_factory):
    before = checkpoint_factory(
        "before.safetensors",
        {"x": np.ones((400,), dtype=np.float32)},
    )
    after = checkpoint_factory(
        "after.safetensors",
        {"x": np.full((400,), 10.0, dtype=np.float32)},
    )

    audit_report = audit_diff_report(
        diff_checkpoints(before, after, use_cache=False),
        fail_on=["norm-spike"],
    )

    finding = next(finding for finding in audit_report.findings if finding.category == "norm-spike")
    assert not audit_report.passed
    assert finding.value == pytest.approx(10.0)
    assert finding.threshold == pytest.approx(10.0)


def test_audit_uses_rms_instead_of_tensor_size(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "scaled.safetensors",
        {
            "small.a": np.ones((1,), dtype=np.float32),
            "small.b": np.ones((1,), dtype=np.float32),
            "small.c": np.ones((1,), dtype=np.float32),
            "large.same_distribution": np.ones((400,), dtype=np.float32),
        },
    )

    report = audit_stats_report(build_stats_report(checkpoint, use_cache=False), ["norm-spike"])

    assert report.passed
    assert all(finding.category != "norm-spike" for finding in report.findings)


def test_audit_flags_true_rms_spike(checkpoint_factory):
    checkpoint = checkpoint_factory(
        "spike.safetensors",
        {
            "small.a": np.ones((1,), dtype=np.float32),
            "small.b": np.ones((1,), dtype=np.float32),
            "small.c": np.ones((1,), dtype=np.float32),
            "large.spike": np.full((400,), 10.0, dtype=np.float32),
        },
    )

    report = audit_stats_report(build_stats_report(checkpoint, use_cache=False), ["norm-spike"])

    assert not report.passed
    finding = next(finding for finding in report.findings if finding.category == "norm-spike")
    assert finding.tensor == "large.spike"
    assert finding.value == pytest.approx(10.0)
    assert finding.threshold == pytest.approx(10.0)


def test_identical_diff_does_not_reload_tensors(checkpoint_factory, monkeypatch):
    checkpoint = checkpoint_factory(
        "same.safetensors",
        {"a": np.ones((2,), dtype=np.float32), "b": np.zeros((3,), dtype=np.float32)},
    )
    calls = 0
    original = SafeTensorsFile.get_tensor

    def counted(self, name):
        nonlocal calls
        calls += 1
        return original(self, name)

    monkeypatch.setattr(SafeTensorsFile, "get_tensor", counted)

    report = diff_checkpoints(checkpoint, checkpoint, use_cache=False)

    assert calls == 0
    assert all(diff.cosine_distance == 0.0 for diff in report.tensors)


def test_chunked_cosine_matches_numpy():
    left = np.linspace(-2.0, 3.0, 1_048_583, dtype=np.float32)
    right = left * np.float32(1.25) + np.float32(0.5)
    expected = 1.0 - float(
        np.dot(left.astype(np.float64), right.astype(np.float64))
        / (np.linalg.norm(left.astype(np.float64)) * np.linalg.norm(right.astype(np.float64)))
    )

    assert _cosine_distance(left, right) == pytest.approx(expected, abs=1e-12)
    assert _cosine_distance(np.zeros(4), np.zeros(4)) == 0.0
    assert _cosine_distance(np.zeros(4), np.ones(4)) == 1.0


def test_combined_audit_prefers_differential_finding(checkpoint_factory):
    baseline = checkpoint_factory("baseline.safetensors", {"x": np.array([1.0], dtype=np.float32)})
    candidate = checkpoint_factory(
        "candidate.safetensors",
        {"x": np.array([np.nan], dtype=np.float32)},
    )
    report = audit_combined_report(diff_checkpoints(baseline, candidate, use_cache=False), ["nan"])
    nan_findings = [finding for finding in report.findings if finding.category == "nan"]

    assert len(nan_findings) == 1
    assert nan_findings[0].message == "Tensor gained NaN values."
