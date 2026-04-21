"""Three-level JSON parse + fuzzy match for VLM classification output."""
from __future__ import annotations

import json
import re
from difflib import get_close_matches
from typing import Any


_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def _extract_first_json_obj(text: str) -> dict | None:
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
    norm = text.lower().replace(" ", "_")
    for name in sorted(class_names, key=len, reverse=True):
        if name in norm:
            return name
    return None


def parse_prediction(raw: str, class_names: list[str]) -> dict[str, Any]:
    """Parse one VLM raw output string into a structured prediction.

    parse_level legend:
        1 — strict JSON parse + known class
        2 — JSON parse + fuzzy match
        3 — no JSON, token scan matched a class substring
        4 — complete failure (pred_idx=-1)
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
