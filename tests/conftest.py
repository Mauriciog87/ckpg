from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file


@pytest.fixture
def checkpoint_factory(tmp_path: Path):
    def create(name: str, tensors: dict[str, np.ndarray]) -> Path:
        path = tmp_path / name
        save_file(tensors, path)
        return path

    return create
