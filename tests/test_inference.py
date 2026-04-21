from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from PIL import Image

from model.inference import VLMClassifier
from utils.config import (
    Config,
    DatasetConfig,
    EvalConfig,
    InferenceConfig,
    ModelConfig,
    PromptConfig,
)


def make_cfg() -> Config:
    return Config(
        model=ModelConfig(path="/tmp/model"),
        dataset=DatasetConfig(name="imagenet", root="/tmp", split="val"),
        inference=InferenceConfig(batch_size=2, temperature=0.0, max_tokens=32),
        prompt=PromptConfig(template="classify_closed_set", inject_all_classes=True),
        eval=EvalConfig(),
    )


class _FakeOutput:
    def __init__(self, text: str) -> None:
        self.outputs = [MagicMock(text=text)]


def test_classify_batch_returns_parsed_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_llm = MagicMock()
    fake_llm.generate.return_value = [
        _FakeOutput('{"class": "goldfish", "confidence": 0.9}'),
        _FakeOutput('{"class": "tench", "confidence": 0.5}'),
    ]
    fake_tokenizer = MagicMock()
    fake_tokenizer.apply_chat_template.return_value = "PROMPT"
    fake_llm.get_tokenizer.return_value = fake_tokenizer

    monkeypatch.setattr("model.inference._build_llm", lambda cfg: fake_llm)

    cls = VLMClassifier(make_cfg())
    imgs = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
    class_names = ["tench", "goldfish"]

    results = cls.classify_batch(imgs, class_names)
    assert len(results) == 2
    assert results[0]["pred_class"] == "goldfish"
    assert results[0]["pred_idx"] == 1
    assert results[1]["pred_class"] == "tench"
    assert results[1]["pred_idx"] == 0
    fake_llm.generate.assert_called_once()
    args, kwargs = fake_llm.generate.call_args
    requests = args[0] if args else kwargs.get("prompts")
    assert len(requests) == 2
