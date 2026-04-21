# Qwen3.5-VL Image Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vLLM-powered zero-shot image classification pipeline using Qwen3.5-2B (multimodal) on ImageNet-1K, with a modular codebase ready to extend to long-tail (LT) experiments and fine-tuning.

**Architecture:** Config-driven Python project. `dataset/` yields `(PIL.Image, label_idx, synset)` tuples via a shared abstract base class. `model/inference.py` wraps `vllm.LLM` for batched multimodal generation and parses JSON output with 3-level fallback. `main.py` orchestrates via YAML config + CLI overrides. Fine-tune and LT modules are placeholders with stable interfaces.

**Tech Stack:** Python 3.12 • vLLM 0.19.0 • PyTorch 2.10 • Qwen3.5-2B (local HF weights) • PyYAML • PIL • scipy (for devkit `meta.mat`) • pytest • difflib (stdlib fuzzy match).

**Environment assumptions (verified before planning):**
- Model path: `/home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`
- ImageNet path: `/opt/ImageNet/{train,val,devkit}`
- GPU: 3× RTX 4090 D (48 GB each); default to single-GPU run
- Project is NOT yet a git repo → Task 1 runs `git init`
- `rapidfuzz` NOT installed → use `difflib` (stdlib)

---

## File Structure

```
Qwen3.5-LT/
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── configs/
│   ├── base.yaml
│   ├── imagenet.yaml
│   └── imagenet_lt.yaml
├── dataset/
│   ├── __init__.py
│   ├── base.py                 # ClassificationDataset abstract base
│   ├── imagenet.py             # ImageNet ImageFolder wrapper + synset→name
│   └── imagenet_lt.py          # Stub subclass for later LT work
├── model/
│   ├── __init__.py
│   ├── inference.py            # VLMClassifier (vLLM offline batch)
│   └── finetune.py             # Placeholder FinetuneRunner (NotImplementedError)
├── prompts/
│   ├── __init__.py
│   ├── templates.py            # SYSTEM_PROMPT, build_user_prompt()
│   └── imagenet_classes.txt    # 1000 canonical ImageNet class names (generated)
├── utils/
│   ├── __init__.py
│   ├── config.py               # YAML → dataclasses
│   ├── logger.py               # Stdlib logging setup
│   ├── metrics.py              # Top-1/Top-5/per-class + parse-level stats
│   └── postprocess.py          # 3-level JSON parse + fuzzy match
├── scripts/
│   ├── build_class_names.py    # One-off: devkit/meta.mat → imagenet_classes.txt
│   ├── run_inference.sh        # Parametric inference script
│   └── run_finetune.sh         # Placeholder
├── tests/
│   ├── __init__.py
│   ├── conftest.py             # Shared fixtures (tiny dataset, dummy config)
│   ├── test_config.py
│   ├── test_prompts.py
│   ├── test_postprocess.py
│   ├── test_metrics.py
│   ├── test_dataset.py
│   ├── test_inference.py       # Mock-based; no real vLLM
│   └── test_smoke.py           # Real vLLM (skip without GPU/model)
├── outputs/                    # Created at runtime
└── docs/
    └── superpowers/
        ├── specs/2026-04-21-qwen-vlm-classification-design.md
        └── plans/2026-04-21-qwen-vlm-classification.md  (this file)
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `/home/kemove/Experiment/Qwen3.5-LT/.gitignore`
- Create: `/home/kemove/Experiment/Qwen3.5-LT/requirements.txt`
- Create: empty `__init__.py` in `dataset/`, `model/`, `prompts/`, `utils/`, `tests/`
- Initialize git repo

- [ ] **Step 1: Initialize git repo**

Run:
```bash
cd /home/kemove/Experiment/Qwen3.5-LT && git init -b main
```
Expected: "Initialized empty Git repository in /home/kemove/Experiment/Qwen3.5-LT/.git/"

- [ ] **Step 2: Write `.gitignore`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
.pytest_cache/
.mypy_cache/
.ruff_cache/
.venv/
venv/

# Project outputs
outputs/
wandb/
*.log

# IDE
.vscode/
.idea/

# OS
.DS_Store
Thumbs.db
```

- [ ] **Step 3: Write `requirements.txt`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/requirements.txt`

```text
# Runtime (already verified in the env; listed for reproducibility)
vllm>=0.19.0
torch>=2.10
torchvision>=0.25
transformers>=4.57.0.dev0
Pillow
PyYAML
tqdm
scipy

# Test
pytest
```

- [ ] **Step 4: Create empty package files**

Run:
```bash
cd /home/kemove/Experiment/Qwen3.5-LT
for pkg in dataset model prompts utils tests; do
  touch "${pkg}/__init__.py"
done
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add .gitignore requirements.txt dataset/__init__.py model/__init__.py prompts/__init__.py utils/__init__.py tests/__init__.py docs/
git commit -m "chore: scaffold project directories and requirements"
```

Expected: commit succeeds, `git status` shows working tree clean (except `main.py`, `README.md` and other pre-existing empty files).

---

## Task 2: Build ImageNet Class Names

**Files:**
- Create: `scripts/build_class_names.py`
- Create: `prompts/imagenet_classes.txt` (generated)

- [ ] **Step 1: Write the generator script**

Path: `/home/kemove/Experiment/Qwen3.5-LT/scripts/build_class_names.py`

```python
"""One-off: derive sorted 1000-class names from ImageNet devkit/meta.mat.

Writes one class name per line to prompts/imagenet_classes.txt, ordered by
synset ID alphabetically — which matches torchvision.ImageFolder ordering on
/opt/ImageNet/train (and val).

Each line is the canonical first phrase from WordNet, e.g. "tench" instead of
"tench, Tinca tinca".
"""
from __future__ import annotations

import argparse
from pathlib import Path

import scipy.io


def load_synset_to_name(meta_mat_path: Path) -> dict[str, str]:
    mat = scipy.io.loadmat(str(meta_mat_path), squeeze_me=True)
    synsets = mat["synsets"]
    mapping: dict[str, str] = {}
    for entry in synsets:
        # entry fields: ILSVRC2012_ID, WNID, words, gloss, num_children, ...
        wnid = str(entry["WNID"])
        words = str(entry["words"])
        if not wnid.startswith("n"):
            # devkit includes hierarchy parents; keep only leaf (n########).
            continue
        # Use the first comma-separated phrase as the canonical name.
        canonical = words.split(",")[0].strip().lower().replace(" ", "_")
        mapping[wnid] = canonical
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta-mat",
        type=Path,
        default=Path("/opt/ImageNet/devkit/data/meta.mat"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prompts/imagenet_classes.txt",
    )
    args = parser.parse_args()

    synset_to_name = load_synset_to_name(args.meta_mat)
    sorted_synsets = sorted(synset_to_name.keys())
    if len(sorted_synsets) != 1000:
        raise ValueError(
            f"Expected 1000 leaf synsets, got {len(sorted_synsets)}. "
            f"Check meta.mat at {args.meta_mat}."
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for synset in sorted_synsets:
            f.write(synset_to_name[synset] + "\n")

    print(f"Wrote {len(sorted_synsets)} class names to {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it to produce the class list**

Run:
```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python scripts/build_class_names.py
```
Expected stdout: `Wrote 1000 class names to /home/kemove/Experiment/Qwen3.5-LT/prompts/imagenet_classes.txt`

- [ ] **Step 3: Verify file**

Run:
```bash
wc -l /home/kemove/Experiment/Qwen3.5-LT/prompts/imagenet_classes.txt
head -5 /home/kemove/Experiment/Qwen3.5-LT/prompts/imagenet_classes.txt
```
Expected: exactly `1000` lines; first 5 are recognizable English ImageNet class names (e.g. `tench`, `goldfish`, `great_white_shark`, `tiger_shark`, `hammerhead`).

- [ ] **Step 4: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add scripts/build_class_names.py prompts/imagenet_classes.txt
git commit -m "feat: generate canonical ImageNet-1K class names from devkit"
```

---

## Task 3: Config Loader + Logger

**Files:**
- Create: `utils/config.py`
- Create: `utils/logger.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loader**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_config.py`

```python
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
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_config.py -v
```
Expected: `ModuleNotFoundError: No module named 'utils.config'` or `ImportError`.

- [ ] **Step 3: Implement `utils/config.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/utils/config.py`

```python
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
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_config.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Write `utils/logger.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/utils/logger.py`

```python
"""Minimal logging helper: console + optional file handler."""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "qwen35_lt", log_file: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured in this process
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
```

- [ ] **Step 6: Smoke-check logger import**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -c "from utils.logger import get_logger; get_logger().info('hello')"
```
Expected: one INFO line printed containing "hello".

- [ ] **Step 7: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add utils/config.py utils/logger.py tests/test_config.py
git commit -m "feat(utils): yaml config loader with dotted overrides + logger"
```

---

## Task 4: Prompt Templates

**Files:**
- Create: `prompts/templates.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write failing tests**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_prompts.py`

```python
from __future__ import annotations

from prompts.templates import SYSTEM_PROMPT, build_user_prompt, load_class_names


def test_system_prompt_contains_json_instruction() -> None:
    assert "JSON" in SYSTEM_PROMPT
    assert "class" in SYSTEM_PROMPT
    assert "confidence" in SYSTEM_PROMPT


def test_build_user_prompt_includes_all_classes() -> None:
    classes = ["cat", "dog", "horse"]
    prompt = build_user_prompt(classes, inject_all=True)
    for name in classes:
        assert name in prompt
    assert "3" in prompt  # n=3


def test_build_user_prompt_without_class_list() -> None:
    prompt = build_user_prompt(["cat", "dog"], inject_all=False)
    assert "cat" not in prompt and "dog" not in prompt
    assert "ImageNet" in prompt or "category" in prompt


def test_load_class_names_from_text_file(tmp_path):
    f = tmp_path / "cls.txt"
    f.write_text("cat\ndog\nhorse\n")
    names = load_class_names(f)
    assert names == ["cat", "dog", "horse"]
```

- [ ] **Step 2: Run, expect ImportError**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_prompts.py -v
```
Expected: ImportError.

- [ ] **Step 3: Implement `prompts/templates.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/prompts/templates.py`

```python
"""Prompt templates and class-name loader for image classification."""
from __future__ import annotations

from pathlib import Path


SYSTEM_PROMPT = (
    "You are an expert image classifier. Given an image, identify the most "
    "likely category. Respond with ONLY a valid JSON object, no extra text, "
    'in the form:\n{"class": "<category name>", "confidence": <float 0-1>}'
)


CLOSED_SET_TEMPLATE = (
    "Classify this image into exactly one of the following {n} ImageNet "
    "categories:\n\n{class_list}\n\nReturn only the JSON object."
)


OPEN_SET_TEMPLATE = (
    "Classify this image. Return the most specific ImageNet category name "
    "you can infer. Return only the JSON object."
)


def build_user_prompt(class_names: list[str], inject_all: bool = True) -> str:
    """Construct the user-side prompt.

    When ``inject_all`` is True, the full candidate list is embedded — this
    constrains the model to a closed-set choice. When False, the model is
    asked to produce a free-form label which downstream fuzzy matching must
    align to the candidate list.
    """
    if inject_all:
        class_list = ", ".join(class_names)
        return CLOSED_SET_TEMPLATE.format(n=len(class_names), class_list=class_list)
    return OPEN_SET_TEMPLATE


def load_class_names(path: Path | str) -> list[str]:
    names = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                names.append(line)
    return names
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_prompts.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add prompts/templates.py tests/test_prompts.py
git commit -m "feat(prompts): templates + class-names loader"
```

---

## Task 5: JSON Post-processing

**Files:**
- Create: `utils/postprocess.py`
- Create: `tests/test_postprocess.py`

- [ ] **Step 1: Write failing tests**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_postprocess.py`

```python
from __future__ import annotations

import pytest

from utils.postprocess import parse_prediction


CLASS_NAMES = ["tench", "goldfish", "great_white_shark", "tiger_shark"]


def test_level1_strict_json() -> None:
    raw = '{"class": "goldfish", "confidence": 0.92}'
    result = parse_prediction(raw, CLASS_NAMES)
    assert result["pred_class"] == "goldfish"
    assert result["pred_idx"] == 1
    assert abs(result["confidence"] - 0.92) < 1e-6
    assert result["parse_level"] == 1


def test_level1_json_with_surrounding_noise() -> None:
    raw = 'Sure! Here is my answer: {"class": "tench", "confidence": 0.5} Thanks.'
    result = parse_prediction(raw, CLASS_NAMES)
    assert result["pred_class"] == "tench"
    assert result["pred_idx"] == 0
    assert result["parse_level"] == 1


def test_level2_json_with_unknown_class_fuzzy_match() -> None:
    # "goldfishh" is close to "goldfish"
    raw = '{"class": "goldfishh", "confidence": 0.8}'
    result = parse_prediction(raw, CLASS_NAMES)
    assert result["pred_class"] == "goldfish"
    assert result["pred_idx"] == 1
    assert result["parse_level"] == 2


def test_level3_no_json_token_match() -> None:
    raw = "I think this is a tiger_shark swimming in water."
    result = parse_prediction(raw, CLASS_NAMES)
    assert result["pred_class"] == "tiger_shark"
    assert result["pred_idx"] == 3
    assert result["parse_level"] == 3


def test_level_fail_no_match() -> None:
    raw = "Blurry picture, cannot tell."
    result = parse_prediction(raw, CLASS_NAMES)
    assert result["pred_idx"] == -1
    assert result["confidence"] == 0.0
    assert result["parse_level"] == 4  # failure sentinel
```

- [ ] **Step 2: Run, expect fail**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_postprocess.py -v
```

- [ ] **Step 3: Implement `utils/postprocess.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/utils/postprocess.py`

```python
"""Three-level JSON parse + fuzzy match for VLM classification output."""
from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_first_json_obj(text: str) -> dict | None:
    """Return the first top-level JSON object dict found in text, or None."""
    for match in _JSON_OBJ_RE.finditer(text):
        candidate = match.group(0)
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


def _fuzzy_match(query: str, class_names: list[str]) -> str | None:
    norm_query = query.strip().lower().replace(" ", "_")
    if norm_query in class_names:
        return norm_query
    matches = get_close_matches(norm_query, class_names, n=1, cutoff=0.6)
    return matches[0] if matches else None


def _token_scan(text: str, class_names: list[str]) -> str | None:
    """Find any class name whose underscored form appears as a substring."""
    norm = text.lower().replace(" ", "_")
    # Prefer longer names first to avoid partial hits (e.g. 'shark' inside 'tiger_shark')
    for name in sorted(class_names, key=len, reverse=True):
        if name in norm:
            return name
    return None


def parse_prediction(raw: str, class_names: list[str]) -> dict[str, Any]:
    """Parse one VLM raw output string into a structured prediction.

    Returns a dict with keys: pred_class, pred_idx, confidence, raw, parse_level.
    parse_level legend: 1=strict JSON+known class, 2=JSON+fuzzy match,
    3=no-JSON token scan, 4=complete failure.
    """
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    obj = _extract_first_json_obj(raw)
    if obj is not None and "class" in obj:
        cls = str(obj["class"]).strip().lower().replace(" ", "_")
        conf = float(obj.get("confidence", 0.0))
        if cls in class_to_idx:
            return {
                "pred_class": cls,
                "pred_idx": class_to_idx[cls],
                "confidence": conf,
                "raw": raw,
                "parse_level": 1,
            }
        fuzzy = _fuzzy_match(cls, class_names)
        if fuzzy is not None:
            return {
                "pred_class": fuzzy,
                "pred_idx": class_to_idx[fuzzy],
                "confidence": conf,
                "raw": raw,
                "parse_level": 2,
            }

    token = _token_scan(raw, class_names)
    if token is not None:
        return {
            "pred_class": token,
            "pred_idx": class_to_idx[token],
            "confidence": 0.0,
            "raw": raw,
            "parse_level": 3,
        }

    return {
        "pred_class": "",
        "pred_idx": -1,
        "confidence": 0.0,
        "raw": raw,
        "parse_level": 4,
    }
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_postprocess.py -v
```
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add utils/postprocess.py tests/test_postprocess.py
git commit -m "feat(utils): 3-level JSON parser with fuzzy and token fallback"
```

---

## Task 6: Metrics

**Files:**
- Create: `utils/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing tests**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_metrics.py`

```python
from __future__ import annotations

from utils.metrics import compute_metrics


def test_compute_metrics_all_correct() -> None:
    preds = [{"pred_idx": 0, "parse_level": 1}, {"pred_idx": 1, "parse_level": 1}]
    gts = [0, 1]
    out = compute_metrics(preds, gts, num_classes=3)
    assert out["top1_acc"] == 1.0
    assert out["parse_level_ratio"]["1"] == 1.0


def test_compute_metrics_half_correct() -> None:
    preds = [
        {"pred_idx": 0, "parse_level": 1},
        {"pred_idx": 2, "parse_level": 2},  # wrong
    ]
    gts = [0, 1]
    out = compute_metrics(preds, gts, num_classes=3)
    assert out["top1_acc"] == 0.5
    assert out["parse_level_ratio"]["1"] == 0.5
    assert out["parse_level_ratio"]["2"] == 0.5


def test_compute_metrics_failure_counts_as_wrong() -> None:
    preds = [{"pred_idx": -1, "parse_level": 4}]
    gts = [0]
    out = compute_metrics(preds, gts, num_classes=3)
    assert out["top1_acc"] == 0.0
    assert out["parse_level_ratio"]["fail"] == 1.0


def test_compute_metrics_per_class() -> None:
    preds = [
        {"pred_idx": 0, "parse_level": 1},
        {"pred_idx": 0, "parse_level": 1},
        {"pred_idx": 1, "parse_level": 1},
    ]
    gts = [0, 0, 1]
    out = compute_metrics(preds, gts, num_classes=3)
    assert out["per_class_acc"]["0"] == 1.0
    assert out["per_class_acc"]["1"] == 1.0
    # Class 2 has no samples → not in map
    assert "2" not in out["per_class_acc"]
```

- [ ] **Step 2: Run, expect fail**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_metrics.py -v
```

- [ ] **Step 3: Implement `utils/metrics.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/utils/metrics.py`

```python
"""Classification metrics including parse-level stats and throughput."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def compute_metrics(
    preds: list[dict[str, Any]],
    gts: list[int],
    num_classes: int,
) -> dict[str, Any]:
    """Compute top-1, per-class, and parse-level breakdowns.

    ``preds`` entries must contain ``pred_idx`` and ``parse_level`` keys.
    Level 4 (complete parse failure) counts as an incorrect prediction.
    Top-5 is not computed here because current prompt emits only top-1; a
    placeholder field is included so downstream consumers have a stable schema.
    """
    if len(preds) != len(gts):
        raise ValueError(f"preds ({len(preds)}) and gts ({len(gts)}) length mismatch")

    if not preds:
        return {
            "top1_acc": 0.0,
            "top5_acc": None,
            "per_class_acc": {},
            "parse_level_ratio": {},
            "num_samples": 0,
        }

    top1_correct = 0
    per_class_correct: dict[int, int] = defaultdict(int)
    per_class_total: dict[int, int] = defaultdict(int)
    parse_counter: Counter[str] = Counter()

    for pred, gt in zip(preds, gts):
        pred_idx = int(pred["pred_idx"])
        level = int(pred["parse_level"])
        per_class_total[gt] += 1
        if pred_idx == gt:
            top1_correct += 1
            per_class_correct[gt] += 1
        if level == 4:
            parse_counter["fail"] += 1
        else:
            parse_counter[str(level)] += 1

    n = len(preds)
    per_class_acc = {
        str(cls): per_class_correct[cls] / per_class_total[cls]
        for cls in per_class_total
    }
    parse_level_ratio = {k: v / n for k, v in parse_counter.items()}

    return {
        "top1_acc": top1_correct / n,
        "top5_acc": None,
        "per_class_acc": per_class_acc,
        "parse_level_ratio": parse_level_ratio,
        "num_samples": n,
        "num_classes": num_classes,
    }
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_metrics.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add utils/metrics.py tests/test_metrics.py
git commit -m "feat(utils): top-1/per-class/parse-level metrics"
```

---

## Task 7: Dataset Layer (Base + ImageNet)

**Files:**
- Create: `dataset/base.py`
- Create: `dataset/imagenet.py`
- Create: `tests/test_dataset.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write shared fixtures**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/conftest.py`

```python
from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture()
def tiny_imagefolder(tmp_path: Path) -> Path:
    """A 2-class ImageFolder with 3 JPEG images per class (synset-named dirs)."""
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
    # IMPORTANT: number of lines should cover the number of synsets in fixture
    p.write_text("first_class\nsecond_class\n")
    return p
```

- [ ] **Step 2: Write failing tests**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_dataset.py`

```python
from __future__ import annotations

from pathlib import Path

from PIL import Image

from dataset.imagenet import ImageNetDataset


def test_dataset_length(tiny_imagefolder: Path, tiny_class_names_file: Path) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",  # empty split → use root directly
        classes_file=tiny_class_names_file,
    )
    assert len(ds) == 6  # 2 classes × 3 images


def test_dataset_getitem_returns_image_label_synset(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    image, label, synset = ds[0]
    assert isinstance(image, Image.Image)
    assert isinstance(label, int)
    assert synset.startswith("n0000000")


def test_dataset_class_names_alignment(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    # synset_to_idx order matches sorted folder names
    assert ds.synset_to_idx == {"n00000001": 0, "n00000002": 1}
    assert ds.class_names == ["first_class", "second_class"]


def test_dataset_corrupt_image_skipped(
    tiny_imagefolder: Path, tiny_class_names_file: Path
) -> None:
    # Truncate one file to 0 bytes → must be skipped, not crash
    bad = next((tiny_imagefolder / "n00000001").glob("*.jpg"))
    bad.write_bytes(b"")
    ds = ImageNetDataset(
        root=str(tiny_imagefolder),
        split="",
        classes_file=tiny_class_names_file,
    )
    assert len(ds) == 5  # one fewer than before
```

- [ ] **Step 3: Run, expect fail**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_dataset.py -v
```

- [ ] **Step 4: Implement `dataset/base.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/dataset/base.py`

```python
"""Abstract base class for classification datasets yielding (image, label, key)."""
from __future__ import annotations

from abc import abstractmethod

from PIL import Image
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """Common interface for image-classification datasets.

    Each item is ``(image, label_idx, key)`` where ``key`` is a dataset-specific
    stable identifier (synset ID for ImageNet, arbitrary string elsewhere).
    Subclasses must populate ``_class_names`` and ``_synset_to_idx`` during init.
    """

    _class_names: list[str] = []
    _synset_to_idx: dict[str, int] = {}

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]: ...

    @property
    def class_names(self) -> list[str]:
        return self._class_names

    @property
    def synset_to_idx(self) -> dict[str, int]:
        return self._synset_to_idx
```

- [ ] **Step 5: Implement `dataset/imagenet.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/dataset/imagenet.py`

```python
"""ImageNet dataset loader (standard ImageFolder layout)."""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from dataset.base import ClassificationDataset
from prompts.templates import load_class_names


_logger = logging.getLogger("qwen35_lt.dataset")


_ALLOWED_EXTS = {".jpeg", ".jpg", ".png"}


class ImageNetDataset(ClassificationDataset):
    """Iterates an ImageNet-style directory.

    Expected layout:
        <root>/<split>/<synset_id>/<image>.JPEG

    ``split`` may be "train", "val", or "" (use <root> directly).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "val",
        classes_file: str | Path = "prompts/imagenet_classes.txt",
    ) -> None:
        self.root = Path(root)
        self.split_dir = self.root / split if split else self.root
        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"ImageNet split dir not found: {self.split_dir}")

        synsets = sorted(
            d.name for d in self.split_dir.iterdir() if d.is_dir()
        )
        self._synset_to_idx = {s: i for i, s in enumerate(synsets)}

        self._samples: list[tuple[Path, int, str]] = []
        for synset in synsets:
            label = self._synset_to_idx[synset]
            cls_dir = self.split_dir / synset
            for p in sorted(cls_dir.iterdir()):
                if p.suffix.lower() not in _ALLOWED_EXTS:
                    continue
                if p.stat().st_size == 0:
                    _logger.warning("skipping empty image file %s", p)
                    continue
                self._samples.append((p, label, synset))

        raw_names = load_class_names(classes_file)
        # Classes file holds the global 1000-name list. The dataset may cover a
        # subset (e.g. in tests) — truncate if the file has more names than
        # folders, or pad with synset IDs as fallback names if it has fewer.
        if len(raw_names) >= len(synsets):
            self._class_names = raw_names[: len(synsets)]
        else:
            self._class_names = raw_names + synsets[len(raw_names):]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[Image.Image, int, str]:
        path, label, synset = self._samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise RuntimeError(f"Failed to read image {path}: {e}") from e
        return image, label, synset
```

- [ ] **Step 6: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_dataset.py -v
```
Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add dataset/base.py dataset/imagenet.py tests/test_dataset.py tests/conftest.py
git commit -m "feat(dataset): classification base + ImageNet ImageFolder loader"
```

---

## Task 8: Placeholder Modules (LT + Finetune)

**Files:**
- Create: `dataset/imagenet_lt.py`
- Create: `model/finetune.py`

- [ ] **Step 1: Write `dataset/imagenet_lt.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/dataset/imagenet_lt.py`

```python
"""Placeholder ImageNet-LT dataset. Stage-2 work.

Planned behavior: inherit from ImageNetDataset, but sub-select samples per a
long-tail index file (e.g. ImageNet-LT official split). For now, this simply
re-exports the balanced class so the config pathway can be wired end-to-end
while we design the LT sampler.
"""
from __future__ import annotations

from dataset.imagenet import ImageNetDataset


class ImageNetLTDataset(ImageNetDataset):
    def __init__(self, *args, split_file: str | None = None, **kwargs) -> None:
        if split_file is not None:
            raise NotImplementedError(
                "ImageNet-LT sampling not implemented yet (Stage 2)."
            )
        super().__init__(*args, **kwargs)
```

- [ ] **Step 2: Write `model/finetune.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/model/finetune.py`

```python
"""Placeholder fine-tune runner. Stage-2 work.

Interface stays stable so main.py can dispatch --mode finetune today and
return a clear error until the real trainer is wired up.
"""
from __future__ import annotations

from typing import Any


class FinetuneRunner:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def train(self) -> None:
        raise NotImplementedError(
            "Fine-tuning is Stage-2 scope. See docs/superpowers/specs/"
            "2026-04-21-qwen-vlm-classification-design.md Section 12."
        )
```

- [ ] **Step 3: Smoke import check**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -c "
from dataset.imagenet_lt import ImageNetLTDataset
from model.finetune import FinetuneRunner
print('placeholders OK')
"
```
Expected: prints `placeholders OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add dataset/imagenet_lt.py model/finetune.py
git commit -m "feat: stub LT dataset and finetune runner (Stage-2 placeholders)"
```

---

## Task 9: Inference Module (mock-tested)

**Files:**
- Create: `model/inference.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing tests (mock-based)**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_inference.py`

```python
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
    # Stub vllm.LLM so we don't load weights in unit tests.
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
    # vllm.LLM.generate invoked exactly once with 2 requests
    fake_llm.generate.assert_called_once()
    args, kwargs = fake_llm.generate.call_args
    requests = args[0] if args else kwargs.get("prompts")
    assert len(requests) == 2
```

- [ ] **Step 2: Run test, expect fail**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_inference.py -v
```

- [ ] **Step 3: Implement `model/inference.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/model/inference.py`

```python
"""VLMClassifier: vLLM offline batch classifier for multimodal Qwen3.5."""
from __future__ import annotations

import logging
from typing import Any

from PIL import Image

from prompts.templates import SYSTEM_PROMPT, build_user_prompt
from utils.config import Config
from utils.postprocess import parse_prediction


_logger = logging.getLogger("qwen35_lt.inference")


def _build_llm(cfg: Config):
    """Isolated import + construction so unit tests can monkeypatch this."""
    from vllm import LLM

    return LLM(
        model=cfg.model.path,
        trust_remote_code=cfg.model.trust_remote_code,
        dtype=cfg.model.dtype,
        gpu_memory_utilization=cfg.model.gpu_memory_utilization,
        max_model_len=cfg.model.max_model_len,
        limit_mm_per_prompt=cfg.model.limit_mm_per_prompt,
    )


def _build_sampling_params(cfg: Config):
    from vllm import SamplingParams

    return SamplingParams(
        temperature=cfg.inference.temperature,
        max_tokens=cfg.inference.max_tokens,
        stop=None,
    )


class VLMClassifier:
    """Thin wrapper around vllm.LLM for image classification via prompts."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._llm = _build_llm(cfg)
        self._tokenizer = self._llm.get_tokenizer()
        self._sampling = _build_sampling_params(cfg)

    def _render_prompt(self, class_names: list[str]) -> str:
        user_text = build_user_prompt(
            class_names, inject_all=self.cfg.prompt.inject_all_classes
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            },
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def classify_batch(
        self,
        images: list[Image.Image],
        class_names: list[str],
    ) -> list[dict[str, Any]]:
        """Run one batch through vLLM and parse outputs."""
        prompt_text = self._render_prompt(class_names)
        requests = [
            {"prompt": prompt_text, "multi_modal_data": {"image": img}}
            for img in images
        ]
        outputs = self._llm.generate(requests, self._sampling)
        results = []
        for out in outputs:
            raw = out.outputs[0].text if out.outputs else ""
            results.append(parse_prediction(raw, class_names))
        return results
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_inference.py -v
```
Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add model/inference.py tests/test_inference.py
git commit -m "feat(model): VLMClassifier wrapping vLLM offline batch inference"
```

---

## Task 10: main.py Entry

**Files:**
- Modify (from empty): `main.py`

- [ ] **Step 1: Implement `main.py`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/main.py`

```python
"""CLI entry for Qwen3.5 image classification evaluation.

Examples:
    python main.py --config configs/imagenet.yaml --num-samples 50
    python main.py --config configs/imagenet.yaml --full
    python main.py --config configs/imagenet.yaml --mode finetune  # placeholder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from tqdm import tqdm

from utils.config import Config, load_config
from utils.logger import get_logger
from utils.metrics import compute_metrics


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", choices=["eval", "finetune"], default="eval")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--num-samples", type=int, default=None)
    g.add_argument("--full", action="store_true")
    p.add_argument("--run-id", type=str, default=None)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def _resolve_run_id(cfg: Config, cli_id: str | None) -> str:
    if cli_id:
        return cli_id
    if cfg.run_id:
        return cfg.run_id
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{cfg.dataset.name}_{ts}"


def _apply_cli_overrides(cfg: Config, args: argparse.Namespace) -> None:
    if args.full:
        cfg.eval.num_samples = None
    elif args.num_samples is not None:
        cfg.eval.num_samples = args.num_samples


def _dump_config_snapshot(cfg: Config, path: Path) -> None:
    def _to_dict(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
        if isinstance(obj, dict):
            return {k: _to_dict(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_dict(v) for v in obj]
        return obj

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(_to_dict(cfg), f, sort_keys=False)


def _build_dataset(cfg: Config):
    name = cfg.dataset.name.lower()
    if name == "imagenet":
        from dataset.imagenet import ImageNetDataset

        return ImageNetDataset(
            root=cfg.dataset.root,
            split=cfg.dataset.split,
            classes_file=cfg.dataset.classes_file,
        )
    if name == "imagenet_lt":
        from dataset.imagenet_lt import ImageNetLTDataset

        return ImageNetLTDataset(
            root=cfg.dataset.root,
            split=cfg.dataset.split,
            classes_file=cfg.dataset.classes_file,
        )
    raise ValueError(f"Unknown dataset: {cfg.dataset.name!r}")


def _already_done_count(pred_path: Path) -> int:
    if not pred_path.exists():
        return 0
    with pred_path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _run_eval(cfg: Config, args: argparse.Namespace) -> int:
    run_id = _resolve_run_id(cfg, args.run_id)
    out_dir = Path(cfg.eval.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(log_file=out_dir / "run.log")
    logger.info("run_id=%s output_dir=%s", run_id, out_dir)

    _dump_config_snapshot(cfg, out_dir / "config.yaml")

    dataset = _build_dataset(cfg)
    total_in_ds = len(dataset)
    limit = cfg.eval.num_samples if cfg.eval.num_samples else total_in_ds
    limit = min(limit, total_in_ds)
    logger.info("dataset size=%d, using %d samples", total_in_ds, limit)

    pred_path = out_dir / "predictions.jsonl"
    corrupt_log = out_dir / "corrupt_images.log"
    failed_parse_log = out_dir / "failed_parses.jsonl"
    start_idx = _already_done_count(pred_path) if args.resume else 0
    if args.resume and start_idx > 0:
        logger.info("resuming from index %d", start_idx)
    elif pred_path.exists():
        pred_path.unlink()

    # Model load time is measured separately to compute pure inference throughput
    t_model_start = time.time()
    from model.inference import VLMClassifier  # imported late for fast --help

    classifier = VLMClassifier(cfg)
    model_load_sec = time.time() - t_model_start
    logger.info("model loaded in %.1fs", model_load_sec)

    batch = cfg.inference.batch_size
    gts_seen: list[int] = []
    preds_seen: list[dict] = []

    t_infer_start = time.time()
    with pred_path.open("a", encoding="utf-8") as fout, \
         corrupt_log.open("a", encoding="utf-8") as cflog:
        for batch_start in tqdm(range(start_idx, limit, batch), desc="eval"):
            batch_end = min(batch_start + batch, limit)
            images, labels, synsets, idxs = [], [], [], []
            for i in range(batch_start, batch_end):
                try:
                    img, lbl, syn = dataset[i]
                except Exception as e:
                    cflog.write(f"{i}\t{e}\n")
                    continue
                images.append(img)
                labels.append(lbl)
                synsets.append(syn)
                idxs.append(i)

            if not images:
                continue

            preds = classifier.classify_batch(images, dataset.class_names)
            for idx, lbl, syn, pred in zip(idxs, labels, synsets, preds):
                gt_class = dataset.class_names[lbl] if lbl < len(dataset.class_names) else str(lbl)
                row = {
                    "idx": idx,
                    "synset": syn,
                    "gt_class": gt_class,
                    "gt_idx": lbl,
                    **pred,
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                gts_seen.append(lbl)
                preds_seen.append(pred)
                if pred["parse_level"] == 4:
                    failed_parse_log.parent.mkdir(parents=True, exist_ok=True)
                    with failed_parse_log.open("a", encoding="utf-8") as fplog:
                        fplog.write(json.dumps(row, ensure_ascii=False) + "\n")

    wall = time.time() - t_infer_start
    num = len(preds_seen) or 1
    metrics = compute_metrics(preds_seen, gts_seen, num_classes=len(dataset.class_names))
    metrics["throughput"] = {
        "total_samples": len(preds_seen),
        "wall_time_sec": round(wall, 2),
        "samples_per_sec": round(len(preds_seen) / wall, 3) if wall > 0 else 0.0,
        "avg_latency_ms_per_sample": round(1000 * wall / num, 2),
        "model_load_sec": round(model_load_sec, 2),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("metrics written. top1=%.4f throughput=%.2f samples/s",
                metrics["top1_acc"], metrics["throughput"]["samples_per_sec"])
    return 0


def _run_finetune(cfg: Config) -> int:
    from model.finetune import FinetuneRunner

    runner = FinetuneRunner(cfg)
    runner.train()  # currently raises NotImplementedError
    return 0


def main() -> int:
    args = _parse_cli()
    cfg = load_config(args.config)
    _apply_cli_overrides(cfg, args)
    if args.mode == "eval":
        return _run_eval(cfg, args)
    return _run_finetune(cfg)


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-test `--help`**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python main.py --help
```
Expected: clean argparse help showing `--config`, `--mode`, `--num-samples`, `--full`, `--run-id`, `--resume`.

- [ ] **Step 3: Verify import graph**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -c "
import main
import model.inference
import dataset.imagenet
import dataset.imagenet_lt
import utils.config
import utils.metrics
import utils.postprocess
import prompts.templates
print('all imports OK')
"
```
Expected: `all imports OK`.

- [ ] **Step 4: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add main.py
git commit -m "feat: main.py orchestrating batched eval with streaming JSONL output"
```

---

## Task 11: Config YAML Files

**Files:**
- Create: `configs/base.yaml`
- Create: `configs/imagenet.yaml`
- Create: `configs/imagenet_lt.yaml`

- [ ] **Step 1: Write `configs/base.yaml`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/configs/base.yaml`

```yaml
# Base / shared fields. Other configs copy-and-override rather than
# inheriting programmatically (keeps the loader simple).
model:
  path: /home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
  trust_remote_code: true
  dtype: bfloat16
  gpu_memory_utilization: 0.85
  max_model_len: 4096
  limit_mm_per_prompt: {image: 1}

inference:
  batch_size: 8
  temperature: 0.0
  max_tokens: 64

prompt:
  template: classify_closed_set
  inject_all_classes: true

eval:
  num_samples: null
  save_predictions: true
  output_dir: outputs/

run_id: null
```

- [ ] **Step 2: Write `configs/imagenet.yaml`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/configs/imagenet.yaml`

```yaml
model:
  path: /home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
  trust_remote_code: true
  dtype: bfloat16
  gpu_memory_utilization: 0.85
  max_model_len: 8192
  limit_mm_per_prompt: {image: 1}

dataset:
  name: imagenet
  root: /opt/ImageNet
  split: val
  classes_file: prompts/imagenet_classes.txt

inference:
  batch_size: 8
  temperature: 0.0
  max_tokens: 64

prompt:
  template: classify_closed_set
  inject_all_classes: true

eval:
  num_samples: null
  save_predictions: true
  output_dir: outputs/

run_id: null
```

- [ ] **Step 3: Write `configs/imagenet_lt.yaml`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/configs/imagenet_lt.yaml`

```yaml
# Stage-2 placeholder: same underlying images, different sampling.
# ImageNetLTDataset currently refuses to run with a split_file — remove the
# stub check when the LT sampler is implemented.
model:
  path: /home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B/snapshots/15852e8c16360a2fea060d615a32b45270f8a8fc
  trust_remote_code: true
  dtype: bfloat16
  gpu_memory_utilization: 0.85
  max_model_len: 8192
  limit_mm_per_prompt: {image: 1}

dataset:
  name: imagenet_lt
  root: /opt/ImageNet
  split: val
  classes_file: prompts/imagenet_classes.txt

inference:
  batch_size: 8
  temperature: 0.0
  max_tokens: 64

prompt:
  template: classify_closed_set
  inject_all_classes: true

eval:
  num_samples: null
  save_predictions: true
  output_dir: outputs/

run_id: null
```

- [ ] **Step 4: Verify YAML parses**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -c "
from utils.config import load_config
for f in ['configs/base.yaml', 'configs/imagenet.yaml', 'configs/imagenet_lt.yaml']:
    try:
        cfg = load_config(f)
        print(f, 'OK -> dataset =', getattr(cfg, 'dataset', None))
    except Exception as e:
        print(f, 'FAIL:', e)
"
```
Expected: `imagenet.yaml` and `imagenet_lt.yaml` load OK. `base.yaml` will fail because it has no `dataset` section — that's intentional (it's for copy-paste, not direct loading). If `base.yaml` fails with a `KeyError: 'dataset'`, that is the expected outcome.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add configs/base.yaml configs/imagenet.yaml configs/imagenet_lt.yaml
git commit -m "feat: YAML configs for imagenet and imagenet_lt"
```

---

## Task 12: Shell Scripts

**Files:**
- Create: `scripts/run_inference.sh`
- Create: `scripts/run_finetune.sh`

- [ ] **Step 1: Write `scripts/run_inference.sh`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/scripts/run_inference.sh`

```bash
#!/usr/bin/env bash
# Run Qwen3.5-2B zero-shot classification on ImageNet.
# Usage:
#   bash scripts/run_inference.sh                               # full val
#   bash scripts/run_inference.sh --num-samples 200             # subset
#   bash scripts/run_inference.sh --config configs/imagenet_lt.yaml --full
#   bash scripts/run_inference.sh --resume --run-id foo
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/imagenet.yaml"
EXTRA_ARGS=()
RUN_ID=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)      CONFIG="$2"; shift 2 ;;
        --num-samples) EXTRA_ARGS+=("--num-samples" "$2"); shift 2 ;;
        --full)        EXTRA_ARGS+=("--full"); shift ;;
        --resume)      EXTRA_ARGS+=("--resume"); shift ;;
        --run-id)      RUN_ID="$2"; shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

if [[ -z "${RUN_ID}" ]]; then
    CFG_NAME="$(basename "${CONFIG}" .yaml)"
    RUN_ID="${CFG_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p outputs
python main.py \
    --config "${CONFIG}" \
    --mode eval \
    --run-id "${RUN_ID}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "outputs/${RUN_ID}.log"
```

- [ ] **Step 2: Write `scripts/run_finetune.sh`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/scripts/run_finetune.sh`

```bash
#!/usr/bin/env bash
# Placeholder: fine-tuning is Stage-2 scope.
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/imagenet.yaml}"
python main.py --config "${CONFIG}" --mode finetune
```

- [ ] **Step 3: Make executable**

```bash
chmod +x /home/kemove/Experiment/Qwen3.5-LT/scripts/run_inference.sh \
         /home/kemove/Experiment/Qwen3.5-LT/scripts/run_finetune.sh
```

- [ ] **Step 4: Verify scripts parse (dry-run with --help)**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && bash -n scripts/run_inference.sh && bash -n scripts/run_finetune.sh && echo "shellcheck-lite OK"
```
Expected: `shellcheck-lite OK`.

- [ ] **Step 5: Commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add scripts/run_inference.sh scripts/run_finetune.sh
git commit -m "feat(scripts): parametric run_inference.sh + finetune placeholder"
```

---

## Task 13: Unit Test Suite Green

**Files:** (no new files; verification only)

- [ ] **Step 1: Run full pytest**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest -v --ignore=tests/test_smoke.py
```
Expected: all tests pass. If any fails, fix the underlying module inline and re-run until green — do NOT proceed.

- [ ] **Step 2: Static syntax compile of every .py**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m compileall -q main.py dataset model prompts utils scripts/build_class_names.py
```
Expected: empty stdout (no errors).

- [ ] **Step 3: Record pytest output to file for the final report**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
python -m pytest -v --ignore=tests/test_smoke.py 2>&1 | tee docs/superpowers/plans/TEST_REPORT_unit.txt
```

- [ ] **Step 4: Commit (only if there were fixes applied during this task)**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git status
# If any fixes were needed: git add <files> && git commit -m "fix: <details>"
# Otherwise skip the commit.
```

---

## Task 14: End-to-End Integration Test (Real Model)

**Files:**
- Create: `tests/test_smoke.py`

This is the verification-before-completion gate. It must run against the real model and real ImageNet images. If the model fails to load on this hardware, mark the task `in_progress`, document the failure, and stop.

- [ ] **Step 1: Write the smoke test**

Path: `/home/kemove/Experiment/Qwen3.5-LT/tests/test_smoke.py`

```python
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
    not MODEL_PATH.exists() or not IMAGENET_VAL.exists()
    or os.environ.get("QWEN35_SKIP_SMOKE") == "1",
    reason="model weights or ImageNet val dir missing, or explicitly skipped",
)
def test_smoke_one_image() -> None:
    from utils.config import load_config
    from model.inference import VLMClassifier
    from prompts.templates import load_class_names

    cfg = load_config("configs/imagenet.yaml")
    classes = load_class_names(cfg.dataset.classes_file)
    assert len(classes) == 1000

    first_synset_dir = sorted(IMAGENET_VAL.iterdir())[0]
    first_image = next(p for p in sorted(first_synset_dir.iterdir())
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    from PIL import Image
    img = Image.open(first_image).convert("RGB")

    clf = VLMClassifier(cfg)
    results = clf.classify_batch([img], classes)
    assert len(results) == 1
    r = results[0]
    assert set(r.keys()) >= {"pred_class", "pred_idx", "confidence", "raw", "parse_level"}
    assert -1 <= r["pred_idx"] <= 999
    assert isinstance(r["raw"], str) and len(r["raw"]) > 0
```

- [ ] **Step 2: Run smoke test**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -m pytest tests/test_smoke.py -v -s 2>&1 | tee docs/superpowers/plans/TEST_REPORT_smoke.txt
```
Expected: 1 passed (will take minutes due to model load). If it fails because of vLLM runtime errors (e.g. OOM, model-arch mismatch), resolve before continuing:
  - OOM → lower `gpu_memory_utilization` or `max_model_len` in `configs/imagenet.yaml`
  - Chat template error → inspect `chat_template.jinja`; adjust `_render_prompt` message structure in `model/inference.py`

- [ ] **Step 3: Run 50-sample subset end-to-end via the shell script**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT && bash scripts/run_inference.sh --num-samples 50 --run-id smoke_subset_50 2>&1 | tee docs/superpowers/plans/TEST_REPORT_subset.txt
```
Expected:
  - Script exits with code 0
  - `outputs/smoke_subset_50/predictions.jsonl` contains 50 lines
  - `outputs/smoke_subset_50/metrics.json` exists with `top1_acc > 0` and `parse_level_ratio["1"] >= 0.5`
  - `outputs/smoke_subset_50/run.log` exists
  - `outputs/smoke_subset_50/config.yaml` exists

- [ ] **Step 4: Verify the subset outputs programmatically**

Run this verification inline (not a committed script):
```bash
cd /home/kemove/Experiment/Qwen3.5-LT && python -c "
import json
from pathlib import Path

run_dir = Path('outputs/smoke_subset_50')
pred_lines = (run_dir/'predictions.jsonl').read_text().strip().split('\n')
assert len(pred_lines) == 50, f'expected 50 lines, got {len(pred_lines)}'
for l in pred_lines:
    json.loads(l)  # each line must parse
m = json.loads((run_dir/'metrics.json').read_text())
assert m['top1_acc'] > 0, f'top1_acc={m[\"top1_acc\"]}'
l1 = m['parse_level_ratio'].get('1', 0.0)
assert l1 >= 0.5, f'parse_level_ratio[1]={l1}'
print(f'SUBSET OK — top1={m[\"top1_acc\"]:.4f} parse_l1={l1:.2f} '
      f'throughput={m[\"throughput\"][\"samples_per_sec\"]:.2f}/s')
" 2>&1 | tee -a docs/superpowers/plans/TEST_REPORT_subset.txt
```
Expected: line starting with `SUBSET OK —`.

- [ ] **Step 5: Commit smoke test + evidence reports**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add tests/test_smoke.py docs/superpowers/plans/TEST_REPORT_*.txt
git commit -m "test: end-to-end smoke + 50-sample subset verification"
```

- [ ] **Step 6: Verify `--resume` semantics**

Re-use the existing `smoke_subset_50` run (produced by Step 3) — it has 50 predictions. Re-run with `--resume --num-samples 50` and verify no duplicate work happens:

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
LINES_BEFORE=$(wc -l < outputs/smoke_subset_50/predictions.jsonl)
bash scripts/run_inference.sh --resume --num-samples 50 --run-id smoke_subset_50 2>&1 \
    | tee docs/superpowers/plans/TEST_REPORT_resume.txt
LINES_AFTER=$(wc -l < outputs/smoke_subset_50/predictions.jsonl)
echo "before=${LINES_BEFORE} after=${LINES_AFTER}"
test "${LINES_BEFORE}" = "${LINES_AFTER}" && echo "RESUME-NOOP OK"
test "${LINES_BEFORE}" = "50" && echo "LINE-COUNT OK"
```
Expected: both `RESUME-NOOP OK` and `LINE-COUNT OK` printed. Resume with nothing new to do must not duplicate rows.

- [ ] **Step 7: Commit resume evidence**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add docs/superpowers/plans/TEST_REPORT_resume.txt
git commit -m "test: verify --resume is idempotent at completion boundary"
```

---

## Task 15: README & Final Verification Report

**Files:**
- Modify (from empty): `README.md`

- [ ] **Step 1: Write `README.md`**

Path: `/home/kemove/Experiment/Qwen3.5-LT/README.md`

```markdown
# Qwen3.5-LT

Zero-shot image classification with **Qwen3.5-2B** (multimodal) via **vLLM** on
**ImageNet-1K**, with modular code ready to extend to long-tail benchmarks and
LoRA fine-tuning.

## Quick start

### Install
```
pip install -r requirements.txt
```

### One-time: build class-name list
```
python scripts/build_class_names.py
```
Produces `prompts/imagenet_classes.txt` (1000 lines).

### Run zero-shot eval
```
bash scripts/run_inference.sh --num-samples 200       # quick subset
bash scripts/run_inference.sh --full                  # full val (50k images)
bash scripts/run_inference.sh --config configs/imagenet_lt.yaml --num-samples 200
```

Outputs land in `outputs/<run_id>/` with `predictions.jsonl`, `metrics.json`,
`config.yaml`, `run.log`.

### Resume after interruption
```
bash scripts/run_inference.sh --resume --run-id <existing_run_id>
```

### Fine-tune (Stage 2, placeholder)
```
bash scripts/run_finetune.sh
```
Currently raises `NotImplementedError`.

## Project layout

See `docs/superpowers/specs/2026-04-21-qwen-vlm-classification-design.md` for
the full design.

## Tests
```
pytest -v --ignore=tests/test_smoke.py   # unit tests (fast)
pytest -v tests/test_smoke.py            # real model smoke (slow, needs GPU)
```
```

- [ ] **Step 2: Assemble final verification report**

Path: `/home/kemove/Experiment/Qwen3.5-LT/docs/superpowers/plans/VERIFICATION_REPORT.md`

```markdown
# Verification Report — Qwen3.5-LT (Stage 1)

Date: YYYY-MM-DD   <!-- filled by the implementer at completion time -->

## Unit tests
See `TEST_REPORT_unit.txt`. Summary:
- test_config.py: X passed
- test_prompts.py: X passed
- test_postprocess.py: X passed
- test_metrics.py: X passed
- test_dataset.py: X passed
- test_inference.py: X passed

## Smoke test (real model)
See `TEST_REPORT_smoke.txt`. Outcome: PASS / SKIP / FAIL (with reason).

## 50-sample subset run
See `TEST_REPORT_subset.txt`. Key numbers:
- Top-1 acc: __
- parse_level_ratio[1]: __
- Throughput: __ samples/sec
- Model load: __ seconds

## Static checks
- `python -m compileall` on all modules: PASS
- Import graph smoke: PASS

## Known issues / follow-ups
- (list anything found during implementation, e.g. chat-template quirks)
```

- [ ] **Step 3: Fill in actual numbers from the test-report files**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
# Edit VERIFICATION_REPORT.md, replacing every '__' placeholder with real
# numbers pulled from TEST_REPORT_subset.txt / metrics.json.
```

- [ ] **Step 4: Final commit**

```bash
cd /home/kemove/Experiment/Qwen3.5-LT
git add README.md docs/superpowers/plans/VERIFICATION_REPORT.md
git commit -m "docs: README + final verification report"
```

- [ ] **Step 5: Confirm completion**

Final confirmation checklist — every item must be a YES:
- [ ] Unit tests all pass (`pytest -v --ignore=tests/test_smoke.py`)
- [ ] Smoke test passes on real model, OR is documented-skipped with a clear reason
- [ ] `bash scripts/run_inference.sh --num-samples 50` succeeds end-to-end
- [ ] `outputs/<run>/metrics.json` exists with `top1_acc > 0` and `parse_level_ratio["1"] >= 0.5`
- [ ] `VERIFICATION_REPORT.md` contains real numbers, not `__`
- [ ] Working tree clean (`git status` shows no pending changes)

Report back to the user with:
- Top-1 accuracy on the 50-sample subset
- Throughput (samples/sec)
- Parse-level distribution
- Any issues that were resolved mid-flight
