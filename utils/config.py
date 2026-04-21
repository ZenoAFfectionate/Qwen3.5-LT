"""YAML → dataclass configuration loader with dotted-path CLI overrides."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    path: str
    trust_remote_code: bool = True
    dtype: str = "bfloat16"
    gpu_memory_utilization: float = 0.85
    max_model_len: int = 4096
    limit_mm_per_prompt: dict = field(default_factory=lambda: {"image": 1})


@dataclass
class DatasetConfig:
    name: str
    root: str
    split: str = "val"
    classes_file: str = "prompts/imagenet_classes.txt"


@dataclass
class InferenceConfig:
    batch_size: int = 8
    temperature: float = 0.0
    max_tokens: int = 64


@dataclass
class PromptConfig:
    template: str = "classify_closed_set"
    inject_all_classes: bool = True


@dataclass
class EvalConfig:
    num_samples: Optional[int] = None
    save_predictions: bool = True
    output_dir: str = "outputs/"


@dataclass
class Config:
    model: ModelConfig
    dataset: DatasetConfig
    inference: InferenceConfig
    prompt: PromptConfig
    eval: EvalConfig
    run_id: Optional[str] = None


def _construct_config(data: dict) -> Config:
    return Config(
        model=ModelConfig(**data["model"]),
        dataset=DatasetConfig(**data["dataset"]),
        inference=InferenceConfig(**data["inference"]),
        prompt=PromptConfig(**data["prompt"]),
        eval=EvalConfig(**data["eval"]),
        run_id=data.get("run_id"),
    )


def _apply_override(cfg: Config, dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    obj: Any = cfg
    for p in parts[:-1]:
        if not hasattr(obj, p):
            raise KeyError(f"Unknown config path segment: {p!r} in {dotted_key!r}")
        obj = getattr(obj, p)
    last = parts[-1]
    if not hasattr(obj, last):
        raise KeyError(f"Unknown config key: {dotted_key!r}")
    setattr(obj, last, value)


def load_config(
    path: Path | str,
    overrides: Optional[dict[str, Any]] = None,
) -> Config:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    cfg = _construct_config(data)
    if overrides:
        for k, v in overrides.items():
            _apply_override(cfg, k, v)
    return cfg
