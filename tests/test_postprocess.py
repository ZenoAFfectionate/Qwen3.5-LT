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
    assert result["parse_level"] == 4
