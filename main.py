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


def _read_predictions(pred_path: Path) -> tuple[list[dict], list[int]]:
    """Re-read a predictions.jsonl into (preds, gts) for end-of-run metrics."""
    if not pred_path.exists():
        return [], []
    preds: list[dict] = []
    gts: list[int] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds.append(
                {
                    "pred_idx": row["pred_idx"],
                    "parse_level": row["parse_level"],
                    "confidence": row.get("confidence", 0.0),
                }
            )
            gts.append(row["gt_idx"])
    return preds, gts


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

    t_model_start = time.time()
    from model.inference import VLMClassifier

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
                gt_class = (
                    dataset.class_names[lbl]
                    if lbl < len(dataset.class_names)
                    else str(lbl)
                )
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
                    with failed_parse_log.open("a", encoding="utf-8") as fplog:
                        fplog.write(json.dumps(row, ensure_ascii=False) + "\n")

    wall = time.time() - t_infer_start
    all_preds, all_gts = _read_predictions(pred_path)
    n = len(all_preds) or 1
    metrics = compute_metrics(all_preds, all_gts, num_classes=len(dataset.class_names))
    new_samples = len(preds_seen)
    metrics["throughput"] = {
        "total_samples": len(all_preds),
        "new_samples_this_run": new_samples,
        "wall_time_sec": round(wall, 2),
        "samples_per_sec": round(new_samples / wall, 3) if wall > 0 and new_samples else 0.0,
        "avg_latency_ms_per_sample": round(1000 * wall / new_samples, 2) if new_samples else 0.0,
        "model_load_sec": round(model_load_sec, 2),
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "metrics written. top1=%.4f throughput=%.2f samples/s",
        metrics["top1_acc"],
        metrics["throughput"]["samples_per_sec"],
    )
    return 0


def _run_finetune(cfg: Config) -> int:
    from model.finetune import FinetuneRunner

    runner = FinetuneRunner(cfg)
    runner.train()
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
