from __future__ import annotations

import numpy as np

from ckptguard.policies.audit_policy import audit_diff_report, audit_stats_report
from ckptguard.stats.diff_stats import diff_checkpoints
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
