from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable
from contextlib import suppress
from pathlib import Path

from ckptguard.errors import CkptGuardError


def paths_equivalent(left: Path | str, right: Path | str) -> bool:
    left_path = Path(left)
    right_path = Path(right)
    try:
        if left_path.exists() and right_path.exists() and os.path.samefile(left_path, right_path):
            return True
    except OSError:
        pass
    try:
        left_value = str(left_path.resolve(strict=False))
        right_value = str(right_path.resolve(strict=False))
    except OSError:
        left_value = os.path.abspath(left_path)
        right_value = os.path.abspath(right_path)
    if os.name == "nt":
        left_value = os.path.normcase(left_value)
        right_value = os.path.normcase(right_value)
    return left_value == right_value


def validate_output_paths(
    outputs: Iterable[Path | str | None],
    protected_paths: Iterable[Path | str] = (),
) -> None:
    output_paths = [Path(path) for path in outputs if path is not None]
    protected = [Path(path) for path in protected_paths]
    for output in output_paths:
        if any(paths_equivalent(output, protected_path) for protected_path in protected):
            raise CkptGuardError(f"Output path conflicts with a protected input: {output}")
    for index, output in enumerate(output_paths):
        for other in output_paths[index + 1 :]:
            if paths_equivalent(output, other):
                raise CkptGuardError(f"Output paths conflict: {output} and {other}")


def write_text_atomic(
    path: Path | str,
    content: str,
    protected_paths: Iterable[Path | str] = (),
) -> None:
    output_path = Path(path)
    validate_output_paths([output_path], protected_paths)
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, output_path)
    except OSError as exc:
        raise CkptGuardError(f"Could not write output file {output_path}: {exc}") from exc
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
