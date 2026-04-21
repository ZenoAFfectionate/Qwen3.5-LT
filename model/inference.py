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
