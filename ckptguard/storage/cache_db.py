from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

from ckptguard.errors import CkptGuardError
from ckptguard.models import SCHEMA_VERSION, FileInfo, StatsReport

CACHE_VERSION = 2


class StatsCache:
    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else Path(".ckptguard") / "cache.sqlite"
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CkptGuardError(
                f"Could not create cache directory {self.path.parent}: {exc}"
            ) from exc
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path, timeout=5.0)
            connection.execute("pragma busy_timeout = 5000")
            connection.execute("pragma journal_mode = WAL")
            connection.execute("pragma synchronous = NORMAL")
            return connection
        except (OSError, sqlite3.Error) as exc:
            if connection is not None:
                connection.close()
            raise self._error(exc) from exc

    def _error(self, exc: Exception) -> CkptGuardError:
        return CkptGuardError(f"Cache database is invalid or unavailable: {self.path}: {exc}")

    def _initialize(self) -> None:
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "create table if not exists stats_cache ("
                    "cache_key text primary key, "
                    "schema_version text not null, "
                    "payload text not null, "
                    "created_at text not null)"
                )
                connection.execute(
                    "delete from stats_cache where schema_version <> ?",
                    (CACHE_VERSION,),
                )
        except sqlite3.Error as exc:
            raise self._error(exc) from exc

    def _key(self, info: FileInfo, content_hash: str) -> str:
        raw = (
            f"{CACHE_VERSION}|{SCHEMA_VERSION}|{info.path}|{info.size_bytes}|"
            f"{info.mtime_ns}|{content_hash}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, info: FileInfo, content_hash: str) -> StatsReport | None:
        key = self._key(info, content_hash)
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "select payload from stats_cache where cache_key = ? and schema_version = ?",
                    (key, CACHE_VERSION),
                ).fetchone()
        except sqlite3.Error as exc:
            raise self._error(exc) from exc
        if row is None:
            return None
        try:
            return StatsReport.model_validate_json(row[0])
        except (TypeError, ValueError):
            try:
                with closing(self._connect()) as connection, connection:
                    connection.execute(
                        "delete from stats_cache where cache_key = ?",
                        (key,),
                    )
            except sqlite3.Error as exc:
                raise self._error(exc) from exc
            return None

    def set(self, info: FileInfo, content_hash: str, report: StatsReport) -> None:
        key = self._key(info, content_hash)
        payload = report.model_dump_json()
        try:
            with closing(self._connect()) as connection, connection:
                connection.execute(
                    "insert or replace into stats_cache "
                    "(cache_key, schema_version, payload, created_at) values (?, ?, ?, ?)",
                    (key, CACHE_VERSION, payload, report.generated_at.isoformat()),
                )
        except sqlite3.Error as exc:
            raise self._error(exc) from exc
