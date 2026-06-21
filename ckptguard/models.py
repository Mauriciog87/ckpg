from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"


def utc_now() -> datetime:
    return datetime.now(UTC)


class CkptGuardModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FileInfo(CkptGuardModel):
    path: str
    size_bytes: int
    mtime_ns: int


class TensorStats(CkptGuardModel):
    name: str
    shape: list[int]
    dtype: str
    numel: int
    min: float | None
    max: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None
    linf_norm: float | None
    zero_ratio: float | None
    nan_count: int
    inf_count: int
    sha256: str


class StatsSummary(CkptGuardModel):
    tensor_count: int
    total_numel: int
    nan_tensors: int
    inf_tensors: int
    zero_tensors: int


class StatsReport(CkptGuardModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    file: FileInfo
    metadata: dict[str, str]
    summary: StatsSummary
    tensors: list[TensorStats]


class TensorDiff(CkptGuardModel):
    name: str
    status: Literal["added", "removed", "changed", "unchanged"]
    before: TensorStats | None
    after: TensorStats | None
    changes: list[str]
    score: float
    norm_delta: float | None
    linf_delta: float | None
    zero_ratio_delta: float | None
    mean_delta: float | None
    std_delta: float | None
    cosine_distance: float | None


class DiffSummary(CkptGuardModel):
    total_tensors: int
    added: int
    removed: int
    changed: int
    unchanged: int


class DiffReport(CkptGuardModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    before_file: FileInfo
    after_file: FileInfo
    summary: DiffSummary
    tensors: list[TensorDiff]


class AuditFinding(CkptGuardModel):
    category: str
    severity: Literal["error", "warning"]
    tensor: str | None
    message: str
    value: float | int | str | None = None
    threshold: float | int | str | None = None


class AuditReport(CkptGuardModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    file: FileInfo | None
    findings: list[AuditFinding]
    fail_on: list[str]
    passed: bool


class HtmlReport(CkptGuardModel):
    schema_version: str = SCHEMA_VERSION
    generated_at: datetime = Field(default_factory=utc_now)
    diff: DiffReport
    audit: AuditReport
