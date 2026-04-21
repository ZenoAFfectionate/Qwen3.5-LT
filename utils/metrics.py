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

    Each entry in ``preds`` must contain ``pred_idx`` and ``parse_level``.
    parse_level == 4 is treated as a complete parse failure and still counts
    as an incorrect prediction. ``top5_acc`` is a placeholder for future
    prompts that emit a top-k list — the current JSON prompt yields top-1 only.
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
            "num_classes": num_classes,
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
