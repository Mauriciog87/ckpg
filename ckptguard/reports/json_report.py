from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel

from ckptguard.reports.output import write_text_atomic


def write_json_report(
    report: BaseModel,
    path: Path | str,
    protected_paths: Iterable[Path | str] = (),
) -> None:
    write_text_atomic(
        path,
        report.model_dump_json(indent=2) + "\n",
        protected_paths=protected_paths,
    )
