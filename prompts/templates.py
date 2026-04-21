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
