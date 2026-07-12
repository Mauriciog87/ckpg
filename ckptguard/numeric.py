from __future__ import annotations

import math
from collections.abc import Iterator

import ml_dtypes
import numpy as np

from ckptguard.errors import CkptGuardError

NUMERIC_CHUNK_ELEMENTS = 1_048_576
BFLOAT16_DTYPE = np.dtype(ml_dtypes.bfloat16)


def is_complex_dtype(dtype: np.dtype | str) -> bool:
    return bool(np.issubdtype(np.dtype(dtype), np.complexfloating))


def is_real_numeric_dtype(dtype: np.dtype | str) -> bool:
    resolved = np.dtype(dtype)
    return bool(
        resolved == BFLOAT16_DTYPE
        or resolved == np.dtype(np.bool_)
        or (np.issubdtype(resolved, np.number) and not np.issubdtype(resolved, np.complexfloating))
    )


def supports_nonfinite(dtype: np.dtype | str) -> bool:
    resolved = np.dtype(dtype)
    return bool(
        resolved == BFLOAT16_DTYPE
        or np.issubdtype(resolved, np.floating)
        or np.issubdtype(resolved, np.complexfloating)
    )


def nonfinite_counts(values: np.ndarray) -> tuple[int, int]:
    if not supports_nonfinite(values.dtype):
        return 0, 0
    comparable = values.astype(np.float32, copy=False) if values.dtype == BFLOAT16_DTYPE else values
    with np.errstate(invalid="ignore"):
        nan_count = int(np.count_nonzero(np.isnan(comparable)))
        inf_count = int(np.count_nonzero(np.isinf(comparable)))
    return nan_count, inf_count


def ensure_supported_dtype(dtype: np.dtype | str) -> None:
    resolved = np.dtype(dtype)
    if (
        resolved == BFLOAT16_DTYPE
        or np.issubdtype(resolved, np.number)
        or resolved == np.dtype(np.bool_)
    ):
        return
    raise CkptGuardError(
        f"Unsupported tensor dtype '{resolved}'. BF16 is supported; FP8 and other "
        "NumPy-incompatible safetensors dtypes are not supported."
    )


def iter_flat_chunks(
    tensor: np.ndarray,
    chunk_elements: int = NUMERIC_CHUNK_ELEMENTS,
) -> Iterator[np.ndarray]:
    if tensor.size == 0:
        return
    iterator = np.nditer(
        tensor,
        flags=["external_loop", "buffered"],
        op_flags=[["readonly"]],
        order="C",
        buffersize=chunk_elements,
    )
    for chunk in iterator:
        yield np.asarray(chunk)


def chunk_l2_norm(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    absolute = np.abs(values)
    scale = float(np.max(absolute))
    if scale == 0.0:
        return 0.0
    scaled = absolute / scale
    return scale * math.sqrt(float(np.dot(scaled, scaled)))


def rms(l2_norm: float | None, numel: int) -> float | None:
    if l2_norm is None or numel <= 0:
        return None
    return l2_norm / math.sqrt(numel)
