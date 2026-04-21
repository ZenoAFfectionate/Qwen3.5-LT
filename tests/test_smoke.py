"""End-to-end smoke test. Loads real vLLM model; skipped if unavailable."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

MODEL_PATH = Path(
    "/home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/"
    "snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc"
)
IMAGENET_VAL = Path("/opt/ImageNet/val")


@pytest.mark.skipif(
    not MODEL_PATH.exists()
    or not IMAGENET_VAL.exists()
    or os.environ.get("QWEN35_SKIP_SMOKE") == "1",
    reason="model weights or ImageNet val dir missing, or explicitly skipped",
)
def test_smoke_one_image() -> None:
    from PIL import Image

    from model.inference import VLMClassifier
    from prompts.templates import load_class_names
    from utils.config import load_config

    cfg = load_config("configs/imagenet.yaml")
    classes = load_class_names(cfg.dataset.classes_file)
    assert len(classes) == 1000

    first_synset_dir = sorted(IMAGENET_VAL.iterdir())[0]
    first_image = next(
        p for p in sorted(first_synset_dir.iterdir())
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    img = Image.open(first_image).convert("RGB")

    clf = VLMClassifier(cfg)
    results = clf.classify_batch([img], classes)
    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) >= {
        "pred_class", "pred_idx", "confidence", "raw", "parse_level"
    }
    assert -1 <= r["pred_idx"] <= 999
    assert isinstance(r["raw"], str) and len(r["raw"]) > 0
