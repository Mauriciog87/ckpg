from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from ckptguard.errors import CkptGuardError
from ckptguard.readers.safetensors_reader import file_info, file_sha256
from ckptguard.stats.tensor_stats import build_stats_report
from ckptguard.storage.cache_db import CACHE_VERSION, StatsCache


def test_cache_reuses_and_invalidates_by_file_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    cache = StatsCache(tmp_path / "cache.sqlite")
    save_file({"x": np.array([1.0], dtype=np.float32)}, checkpoint)

    first = build_stats_report(checkpoint, cache=cache)
    second = build_stats_report(checkpoint, cache=cache)

    assert first.generated_at == second.generated_at

    save_file({"x": np.array([2.0, 3.0], dtype=np.float32)}, checkpoint)
    third = build_stats_report(checkpoint, cache=cache)

    assert third.summary.total_numel == 2


def test_cache_invalidates_when_content_changes_with_same_metadata(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    cache = StatsCache(tmp_path / "cache.sqlite")
    timestamp = 1_700_000_000_000_000_000
    save_file({"x": np.array([1.0, 2.0], dtype=np.float32)}, checkpoint)
    os.utime(checkpoint, ns=(timestamp, timestamp))

    first = build_stats_report(checkpoint, cache=cache)
    original_size = checkpoint.stat().st_size

    save_file({"x": np.array([np.nan, 2.0], dtype=np.float32)}, checkpoint)
    os.utime(checkpoint, ns=(timestamp, timestamp))
    second = build_stats_report(checkpoint, cache=cache)

    assert checkpoint.stat().st_size == original_size
    assert first.tensors[0].nan_count == 0
    assert second.tensors[0].nan_count == 1


def test_no_cache_does_not_write_rows(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    cache_path = tmp_path / "cache.sqlite"
    cache = StatsCache(cache_path)
    save_file({"x": np.array([1.0], dtype=np.float32)}, checkpoint)

    build_stats_report(checkpoint, cache=cache, use_cache=False)

    with closing(sqlite3.connect(cache_path)) as connection:
        row_count = connection.execute("select count(*) from stats_cache").fetchone()[0]

    assert row_count == 0


def test_corrupt_cache_returns_domain_error(tmp_path: Path):
    cache_path = tmp_path / "cache.sqlite"
    cache_path.write_bytes(b"not a sqlite database")

    with pytest.raises(CkptGuardError, match="Cache database is invalid or unavailable"):
        StatsCache(cache_path)


def test_invalid_cached_payload_is_deleted_and_recomputed(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    cache_path = tmp_path / "cache.sqlite"
    save_file({"x": np.array([1.0], dtype=np.float32)}, checkpoint)
    cache = StatsCache(cache_path)
    info = file_info(checkpoint)
    content_hash = file_sha256(checkpoint)
    cache_key = cache._key(info, content_hash)

    with closing(sqlite3.connect(cache_path)) as connection, connection:
        connection.execute(
            "insert into stats_cache (cache_key, schema_version, payload, created_at) "
            "values (?, ?, ?, ?)",
            (cache_key, CACHE_VERSION, "not-json", "now"),
        )

    report = build_stats_report(checkpoint, cache=cache)

    assert report.summary.tensor_count == 1
    with closing(sqlite3.connect(cache_path)) as connection:
        payload = connection.execute(
            "select payload from stats_cache where cache_key = ?",
            (cache_key,),
        ).fetchone()[0]
    assert payload != "not-json"


def test_cache_initialization_removes_legacy_rows(tmp_path: Path):
    cache_path = tmp_path / "cache.sqlite"
    with closing(sqlite3.connect(cache_path)) as connection, connection:
        connection.execute(
            "create table stats_cache (cache_key text primary key, schema_version text not null, "
            "payload text not null, created_at text not null)"
        )
        connection.execute(
            "insert into stats_cache values (?, ?, ?, ?)",
            ("legacy", "1", "{}", "now"),
        )

    StatsCache(cache_path)

    with closing(sqlite3.connect(cache_path)) as connection:
        count = connection.execute("select count(*) from stats_cache").fetchone()[0]
    assert count == 0
