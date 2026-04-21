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
        {"pred_idx": 2, "parse_level": 2},
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
    assert "2" not in out["per_class_acc"]
