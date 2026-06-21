from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from ckptguard.stats.tensor_stats import build_stats_report
from ckptguard.storage.cache_db import StatsCache


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


def test_no_cache_does_not_write_rows(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    cache_path = tmp_path / "cache.sqlite"
    cache = StatsCache(cache_path)
    save_file({"x": np.array([1.0], dtype=np.float32)}, checkpoint)

    build_stats_report(checkpoint, cache=cache, use_cache=False)

    with sqlite3.connect(cache_path) as connection:
        row_count = connection.execute("select count(*) from stats_cache").fetchone()[0]

    assert row_count == 0
