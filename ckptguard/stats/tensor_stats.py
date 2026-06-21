from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

from ckptguard.models import StatsReport, StatsSummary, TensorStats
from ckptguard.readers.safetensors_reader import SafeTensorsFile, file_info
from ckptguard.storage.cache_db import StatsCache


def _safe_float(value: object) -> float | None:
    result = float(value)
    if math.isfinite(result):
        return result
    return None


def _hash_tensor(tensor: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(tensor)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _nonfinite_counts(tensor: np.ndarray) -> tuple[int, int]:
    if not (
        np.issubdtype(tensor.dtype, np.floating)
        or np.issubdtype(tensor.dtype, np.complexfloating)
    ):
        return 0, 0
    return int(np.isnan(tensor).sum()), int(np.isinf(tensor).sum())


def _zero_ratio(tensor: np.ndarray) -> float | None:
    if tensor.size == 0:
        return None
    return _safe_float(np.count_nonzero(tensor == 0) / tensor.size)


def calculate_tensor_stats(name: str, tensor: np.ndarray) -> TensorStats:
    nan_count, inf_count = _nonfinite_counts(tensor)
    shape = [int(dim) for dim in tensor.shape]
    dtype = str(tensor.dtype)
    zero_ratio = _zero_ratio(tensor)
    empty_or_complex = tensor.size == 0 or np.issubdtype(tensor.dtype, np.complexfloating)
    has_nonfinite = nan_count > 0 or inf_count > 0
    numeric = np.issubdtype(tensor.dtype, np.number) or np.issubdtype(tensor.dtype, np.bool_)

    values: np.ndarray | None = None
    if numeric and not empty_or_complex and not has_nonfinite:
        values = tensor.astype(np.float64, copy=False)

    return TensorStats(
        name=name,
        shape=shape,
        dtype=dtype,
        numel=int(tensor.size),
        min=_safe_float(np.min(values)) if values is not None else None,
        max=_safe_float(np.max(values)) if values is not None else None,
        mean=_safe_float(np.mean(values)) if values is not None else None,
        std=_safe_float(np.std(values)) if values is not None else None,
        l2_norm=_safe_float(np.linalg.norm(values.ravel(), ord=2)) if values is not None else None,
        linf_norm=(
            _safe_float(np.linalg.norm(values.ravel(), ord=np.inf))
            if values is not None
            else None
        ),
        zero_ratio=zero_ratio,
        nan_count=nan_count,
        inf_count=inf_count,
        sha256=_hash_tensor(tensor),
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
    if cache is not None and use_cache:
        cached = cache.get(info)
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

    if cache is not None and use_cache:
        cache.set(info, report)

    return report
