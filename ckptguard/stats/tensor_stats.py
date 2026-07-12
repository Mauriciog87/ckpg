from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ckptguard.models import StatsReport, StatsSummary, TensorStats
from ckptguard.numeric import (
    chunk_l2_norm,
    ensure_supported_dtype,
    is_complex_dtype,
    is_real_numeric_dtype,
    iter_flat_chunks,
    nonfinite_counts,
)
from ckptguard.readers.safetensors_reader import SafeTensorsFile, file_info, file_sha256
from ckptguard.storage.cache_db import StatsCache


def _safe_float(value: object) -> float | None:
    result = float(value)
    if math.isfinite(result):
        return result
    return None


def calculate_tensor_stats(name: str, tensor: np.ndarray) -> TensorStats:
    ensure_supported_dtype(tensor.dtype)
    shape = [int(dim) for dim in tensor.shape]
    dtype = str(tensor.dtype)
    digest = hashlib.sha256()
    nan_count = 0
    inf_count = 0
    zero_count = 0
    count = 0
    mean = 0.0
    moment = 0.0
    minimum: float | None = None
    maximum: float | None = None
    l2_norm = 0.0
    linf_norm = 0.0
    real_numeric = is_real_numeric_dtype(tensor.dtype)
    complex_dtype = is_complex_dtype(tensor.dtype)
    has_nonfinite = False

    for chunk in iter_flat_chunks(tensor):
        contiguous = np.ascontiguousarray(chunk)
        digest.update(contiguous.tobytes(order="C"))
        zero_count += int(np.count_nonzero(chunk == 0))
        chunk_nan_count, chunk_inf_count = nonfinite_counts(chunk)
        nan_count += chunk_nan_count
        inf_count += chunk_inf_count
        chunk_has_nonfinite = chunk_nan_count > 0 or chunk_inf_count > 0
        has_nonfinite = has_nonfinite or chunk_has_nonfinite
        if not real_numeric or chunk_has_nonfinite:
            continue

        values = chunk.astype(np.float64, copy=False)
        chunk_count = int(values.size)
        chunk_mean = float(np.mean(values))
        deviations = values - chunk_mean
        chunk_moment = float(np.dot(deviations, deviations))
        new_count = count + chunk_count
        delta = chunk_mean - mean
        moment += chunk_moment + delta * delta * count * chunk_count / new_count
        mean += delta * chunk_count / new_count
        count = new_count
        chunk_minimum = float(np.min(values))
        chunk_maximum = float(np.max(values))
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
        l2_norm = math.hypot(l2_norm, chunk_l2_norm(values))
        linf_norm = max(linf_norm, float(np.max(np.abs(values))))

    valid_metrics = real_numeric and not complex_dtype and tensor.size > 0 and not has_nonfinite
    zero_ratio = _safe_float(zero_count / tensor.size) if tensor.size > 0 else None

    return TensorStats(
        name=name,
        shape=shape,
        dtype=dtype,
        numel=int(tensor.size),
        min=_safe_float(minimum) if valid_metrics and minimum is not None else None,
        max=_safe_float(maximum) if valid_metrics and maximum is not None else None,
        mean=_safe_float(mean) if valid_metrics else None,
        std=_safe_float(math.sqrt(max(moment / count, 0.0))) if valid_metrics else None,
        l2_norm=_safe_float(l2_norm) if valid_metrics else None,
        linf_norm=_safe_float(linf_norm) if valid_metrics else None,
        zero_ratio=zero_ratio,
        nan_count=nan_count,
        inf_count=inf_count,
        sha256=digest.hexdigest(),
    )


def _summary(tensors: list[TensorStats]) -> StatsSummary:
    return StatsSummary(
        tensor_count=len(tensors),
        total_numel=sum(tensor.numel for tensor in tensors),
        nan_tensors=sum(1 for tensor in tensors if tensor.nan_count > 0),
        inf_tensors=sum(1 for tensor in tensors if tensor.inf_count > 0),
        zero_tensors=sum(1 for tensor in tensors if tensor.zero_ratio == 1.0),
    )


def build_stats_report(
    path: Path | str,
    cache: StatsCache | None = None,
    use_cache: bool = True,
) -> StatsReport:
    info = file_info(path)
    content_hash: str | None = None
    if cache is not None and use_cache:
        content_hash = file_sha256(info.path)
        cached = cache.get(info, content_hash)
        if cached is not None:
            return cached

    reader = SafeTensorsFile(info.path)
    tensors = [calculate_tensor_stats(name, tensor) for name, tensor in reader.iter_tensors()]
    report = StatsReport(
        file=info,
        metadata=reader.metadata(),
        summary=_summary(tensors),
        tensors=tensors,
    )

    if cache is not None and use_cache and content_hash is not None:
        cache.set(info, content_hash, report)

    return report
