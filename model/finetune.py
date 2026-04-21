"""Placeholder fine-tune runner. Stage-2 work.

Interface stays stable so main.py can dispatch ``--mode finetune`` today and
return a clear error until the real trainer is wired up.
"""
from __future__ import annotations

from typing import Any


class FinetuneRunner:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def train(self) -> None:
        raise NotImplementedError(
            "Fine-tuning is Stage-2 scope. See "
            "docs/superpowers/specs/2026-04-21-qwen-vlm-classification-design.md "
            "Section 12."
        )
