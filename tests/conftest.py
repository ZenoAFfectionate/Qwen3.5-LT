from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tiny_imagefolder(tmp_path: Path) -> Path:
    root = tmp_path / "imgnet_tiny"
    classes = ["n00000001", "n00000002"]
    for cls in classes:
        d = root / cls
        d.mkdir(parents=True)
        for i in range(3):
            img = Image.new("RGB", (8, 8), color=(i * 30, 0, 0))
            img.save(d / f"img_{i}.jpg")
    return root


@pytest.fixture()
def tiny_class_names_file(tmp_path: Path) -> Path:
    p = tmp_path / "cls.txt"
    p.write_text("first_class\nsecond_class\n")
    return p
