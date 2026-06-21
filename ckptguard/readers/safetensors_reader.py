from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
from safetensors import safe_open

from ckptguard.errors import CkptGuardError
from ckptguard.models import FileInfo


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


class SafeTensorsFile:
    def __init__(self, path: Path | str) -> None:
        self.path = ensure_safetensors_path(path)

    def metadata(self) -> dict[str, str]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                return dict(handle.metadata() or {})
        except Exception as exc:
            raise CkptGuardError(
                f"Could not read safetensors metadata from {self.path}: {exc}"
            ) from exc

    def keys(self) -> list[str]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                keys = handle.keys()
                return sorted(str(key) for key in keys)
        except Exception as exc:
            raise CkptGuardError(f"Could not list tensors in {self.path}: {exc}") from exc

    def iter_tensors(self) -> Iterator[tuple[str, np.ndarray]]:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                for key in sorted(handle.keys()):
                    yield str(key), handle.get_tensor(key)
        except Exception as exc:
            raise CkptGuardError(f"Could not read tensors from {self.path}: {exc}") from exc

    def get_tensor(self, name: str) -> np.ndarray:
        try:
            with safe_open(str(self.path), framework="np") as handle:
                return handle.get_tensor(name)
        except Exception as exc:
            raise CkptGuardError(f"Could not read tensor '{name}' from {self.path}: {exc}") from exc
