from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

from ckptguard.models import SCHEMA_VERSION, FileInfo, StatsReport


class StatsCache:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(".ckptguard") / "cache.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "create table if not exists stats_cache ("
                "cache_key text primary key, "
                "schema_version text not null, "
                "payload text not null, "
                "created_at text not null)"
            )

    def _key(self, info: FileInfo) -> str:
        raw = f"{SCHEMA_VERSION}|{info.path}|{info.size_bytes}|{info.mtime_ns}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, info: FileInfo) -> StatsReport | None:
        key = self._key(info)
        with self._connect() as connection:
            row = connection.execute(
                "select payload from stats_cache where cache_key = ? and schema_version = ?",
                (key, SCHEMA_VERSION),
            ).fetchone()
        if row is None:
            return None
        return StatsReport.model_validate_json(row[0])

    def set(self, info: FileInfo, report: StatsReport) -> None:
        key = self._key(info)
        payload = report.model_dump_json()
        with self._connect() as connection:
            connection.execute(
                "insert or replace into stats_cache "
                "(cache_key, schema_version, payload, created_at) values (?, ?, ?, ?)",
                (key, SCHEMA_VERSION, payload, report.generated_at.isoformat()),
            )
