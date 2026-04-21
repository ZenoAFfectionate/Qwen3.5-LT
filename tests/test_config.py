from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from utils.config import Config, load_config


@pytest.fixture()
def minimal_yaml(tmp_path: Path) -> Path:
    data = {
        "model": {
            "path": "/tmp/model",
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.85,
            "max_model_len": 4096,
            "limit_mm_per_prompt": {"image": 1},
        },
        "dataset": {
            "name": "imagenet",
            "root": "/tmp/imagenet",
            "split": "val",
            "classes_file": "prompts/imagenet_classes.txt",
        },
        "inference": {"batch_size": 8, "temperature": 0.0, "max_tokens": 64},
        "prompt": {"template": "classify_closed_set", "inject_all_classes": True},
        "eval": {"num_samples": None, "save_predictions": True, "output_dir": "outputs/"},
        "run_id": None,
    }
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_load_config_returns_dataclass(minimal_yaml: Path) -> None:
    cfg = load_config(minimal_yaml)
    assert isinstance(cfg, Config)
    assert cfg.model.path == "/tmp/model"
    assert cfg.inference.batch_size == 8
    assert cfg.eval.num_samples is None


def test_cli_override_num_samples(minimal_yaml: Path) -> None:
    cfg = load_config(minimal_yaml, overrides={"eval.num_samples": 50})
    assert cfg.eval.num_samples == 50


def test_cli_override_unknown_key_raises(minimal_yaml: Path) -> None:
    with pytest.raises(KeyError):
        load_config(minimal_yaml, overrides={"nonexistent.key": 1})
