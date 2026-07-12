from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ckptguard.models import DiffReport, DiffSummary, TensorDiff, TensorStats
from ckptguard.numeric import (
    NUMERIC_CHUNK_ELEMENTS,
    chunk_l2_norm,
    ensure_supported_dtype,
    is_real_numeric_dtype,
    rms,
)
from ckptguard.readers.safetensors_reader import SafeTensorsFile
from ckptguard.stats.tensor_stats import build_stats_report
from ckptguard.storage.cache_db import StatsCache


def _delta(before: float | None, after: float | None) -> float | None:
    if before is None or after is None:
        return None
    result = after - before
    if math.isfinite(result):
        return result
    return None


def _is_numeric(stats: TensorStats) -> bool:
    return is_real_numeric_dtype(stats.dtype)


def _cosine_distance(before: np.ndarray, after: np.ndarray) -> float | None:
    if before.shape != after.shape or before.size == 0:
        return None
    ensure_supported_dtype(before.dtype)
    ensure_supported_dtype(after.dtype)
    if not (is_real_numeric_dtype(before.dtype) and is_real_numeric_dtype(after.dtype)):
        return None

    left = before.reshape(-1)
    right = after.reshape(-1)
    dot_product = 0.0
    left_norm = 0.0
    right_norm = 0.0
    finite_count = 0
    for start in range(0, before.size, NUMERIC_CHUNK_ELEMENTS):
        stop = min(start + NUMERIC_CHUNK_ELEMENTS, before.size)
        left_chunk = left[start:stop].astype(np.float64, copy=False)
        right_chunk = right[start:stop].astype(np.float64, copy=False)
        finite = np.isfinite(left_chunk) & np.isfinite(right_chunk)
        if not bool(finite.any()):
            continue
        left_finite = left_chunk[finite]
        right_finite = right_chunk[finite]
        finite_count += int(left_finite.size)
        dot_product += float(np.dot(left_finite, right_finite))
        left_norm = math.hypot(left_norm, chunk_l2_norm(left_finite))
        right_norm = math.hypot(right_norm, chunk_l2_norm(right_finite))

    if finite_count == 0:
        return None
    if left_norm == 0.0 and right_norm == 0.0:
        return 0.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0

    similarity = dot_product / (left_norm * right_norm)
    similarity = max(-1.0, min(1.0, similarity))
    distance = 1.0 - similarity
    if math.isfinite(distance):
        return distance
    return None


def _score(
    status: str,
    before: TensorStats | None,
    after: TensorStats | None,
    changes: list[str],
    relative_rms_change: float | None,
    cosine_distance: float | None,
) -> float:
    score = 0.0
    if status in {"added", "removed"}:
        score += 800.0
    if "shape" in changes:
        score += 1000.0
    if "dtype" in changes:
        score += 100.0
    if before is not None and after is not None:
        score += abs(after.nan_count - before.nan_count) * 500.0
        score += abs(after.inf_count - before.inf_count) * 500.0
    if relative_rms_change is not None:
        score += min(abs(relative_rms_change) * 100.0, 1000.0)
    if cosine_distance is not None:
        score += cosine_distance * 100.0
    return score


def _relative_rms_change(before: TensorStats, after: TensorStats) -> float | None:
    before_rms = rms(before.l2_norm, before.numel)
    after_rms = rms(after.l2_norm, after.numel)
    if before_rms is None or after_rms is None:
        return None
    if before_rms == 0.0:
        return 0.0 if after_rms == 0.0 else 1.0
    result = (after_rms - before_rms) / before_rms
    return result if math.isfinite(result) else None


def _matching_diff(
    name: str,
    before: TensorStats,
    after: TensorStats,
    before_file: SafeTensorsFile,
    after_file: SafeTensorsFile,
) -> TensorDiff:
    changes: list[str] = []
    if before.shape != after.shape:
        changes.append("shape")
    if before.dtype != after.dtype:
        changes.append("dtype")
    if before.sha256 != after.sha256:
        changes.append("values")
    if before.nan_count != after.nan_count:
        changes.append("nan")
    if before.inf_count != after.inf_count:
        changes.append("inf")

    norm_delta = _delta(before.l2_norm, after.l2_norm)
    linf_delta = _delta(before.linf_norm, after.linf_norm)
    zero_ratio_delta = _delta(before.zero_ratio, after.zero_ratio)
    mean_delta = _delta(before.mean, after.mean)
    std_delta = _delta(before.std, after.std)
    relative_rms_change = _relative_rms_change(before, after)
    cosine_distance = None

    if (
        before.shape == after.shape
        and before.dtype == after.dtype
        and before.sha256 == after.sha256
    ):
        cosine_distance = 0.0
    elif before.shape == after.shape and _is_numeric(before) and _is_numeric(after):
        cosine_distance = _cosine_distance(
            before_file.get_tensor(name),
            after_file.get_tensor(name),
        )

    status = "changed" if changes else "unchanged"
    return TensorDiff(
        name=name,
        status=status,
        before=before,
        after=after,
        changes=changes,
        score=_score(status, before, after, changes, relative_rms_change, cosine_distance),
        norm_delta=norm_delta,
        linf_delta=linf_delta,
        zero_ratio_delta=zero_ratio_delta,
        mean_delta=mean_delta,
        std_delta=std_delta,
        cosine_distance=cosine_distance,
    )


def diff_checkpoints(
    before_path: Path | str,
    after_path: Path | str,
    cache: StatsCache | None = None,
    use_cache: bool = True,
) -> DiffReport:
    before_report = build_stats_report(before_path, cache=cache, use_cache=use_cache)
    after_report = build_stats_report(after_path, cache=cache, use_cache=use_cache)
    before_by_name = {tensor.name: tensor for tensor in before_report.tensors}
    after_by_name = {tensor.name: tensor for tensor in after_report.tensors}
    before_file = SafeTensorsFile(before_report.file.path)
    after_file = SafeTensorsFile(after_report.file.path)
    diffs: list[TensorDiff] = []

    for name in sorted(before_by_name.keys() | after_by_name.keys()):
        before = before_by_name.get(name)
        after = after_by_name.get(name)
        if before is None and after is not None:
            diffs.append(
                TensorDiff(
                    name=name,
                    status="added",
                    before=None,
                    after=after,
                    changes=["added"],
                    score=_score("added", None, after, ["added"], None, None),
                    norm_delta=None,
                    linf_delta=None,
                    zero_ratio_delta=None,
                    mean_delta=None,
                    std_delta=None,
                    cosine_distance=None,
                )
            )
        elif before is not None and after is None:
            diffs.append(
                TensorDiff(
                    name=name,
                    status="removed",
                    before=before,
                    after=None,
                    changes=["removed"],
                    score=_score("removed", before, None, ["removed"], None, None),
                    norm_delta=None,
                    linf_delta=None,
                    zero_ratio_delta=None,
                    mean_delta=None,
                    std_delta=None,
                    cosine_distance=None,
                )
            )
        elif before is not None and after is not None:
            diffs.append(_matching_diff(name, before, after, before_file, after_file))

    diffs.sort(key=lambda diff: (-diff.score, diff.name))
    summary = DiffSummary(
        total_tensors=len(diffs),
        added=sum(1 for diff in diffs if diff.status == "added"),
        removed=sum(1 for diff in diffs if diff.status == "removed"),
        changed=sum(1 for diff in diffs if diff.status == "changed"),
        unchanged=sum(1 for diff in diffs if diff.status == "unchanged"),
    )
    return DiffReport(
        before_file=before_report.file,
        after_file=after_report.file,
        summary=summary,
        tensors=diffs,
    )
