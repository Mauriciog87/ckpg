from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ckptguard.errors import CkptGuardError
from ckptguard.models import FileInfo
from ckptguard.numeric import ensure_supported_dtype

FILE_HASH_CHUNK_BYTES = 8 * 1024 * 1024


def _read_error(action: str, path: Path, exc: Exception) -> CkptGuardError:
    message = str(exc)
    lowered = message.lower()
    if (
        "float8" in lowered
        or "f8_" in lowered
        or "f8e" in lowered
        or ("numpy" in lowered and "has no attribute" in lowered)
    ):
        return CkptGuardError(
            "Unsupported safetensors dtype. BF16 is supported; FP8 is not supported "
            "by the NumPy backend."
        )
    return CkptGuardError(f"Could not {action} {path}: {message}")


def ensure_safetensors_path(path: Path | str) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".safetensors":
        raise CkptGuardError(f"Only .safetensors files are supported: {checkpoint_path}")
    if not checkpoint_path.exists():
        raise CkptGuardError(f"Checkpoint file does not exist: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise CkptGuardError(f"Checkpoint path is not a file: {checkpoint_path}")
    return checkpoint_path.resolve()


def file_info(path: Path | str) -> FileInfo:
    checkpoint_path = ensure_safetensors_path(path)
    stat = checkpoint_path.stat()
    return FileInfo(
        path=str(checkpoint_path),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
    )


def file_sha256(path: Path | str) -> str:
    checkpoint_path = ensure_safetensors_path(path)
    digest = hashlib.sha256()
    try:
        with checkpoint_path.open("rb") as checkpoint:
            while chunk := checkpoint.read(FILE_HASH_CHUNK_BYTES):
                digest.update(chunk)
    except OSError as exc:
        raise CkptGuardError(f"Could not hash checkpoint file {checkpoint_path}: {exc}") from exc
    return digest.hexdigest()


class SafeTensorsFile:
    def __init__(self, path: Path | str) -> None:
        self.path = ensure_safetensors_path(path)

    def metadata(self) -> dict[str, str]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                return dict(handle.metadata() or {})
        except Exception as exc:
            raise _read_error("read safetensors metadata from", self.path, exc) from exc

    def keys(self) -> list[str]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                keys = handle.keys()
                return sorted(str(key) for key in keys)
        except Exception as exc:
            raise _read_error("list tensors in", self.path, exc) from exc

    def iter_tensors(self) -> Iterator[tuple[str, np.ndarray]]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                for key in sorted(handle.keys()):
                    tensor = handle.get_tensor(key)
                    ensure_supported_dtype(tensor.dtype)
                    yield str(key), tensor
        except CkptGuardError:
            raise
        except Exception as exc:
            raise _read_error("read tensors from", self.path, exc) from exc

    def get_tensor(self, name: str) -> np.ndarray:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                tensor = handle.get_tensor(name)
                ensure_supported_dtype(tensor.dtype)
                return tensor
        except CkptGuardError:
            raise
        except Exception as exc:
            raise _read_error(f"read tensor '{name}' from", self.path, exc) from exc
