from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


def write_json_report(report: BaseModel, path: Path | str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
