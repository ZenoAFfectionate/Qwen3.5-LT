# Verification Report — Qwen3.5-LT (Stage 1)

**Date:** 2026-04-21
**Model:** Qwen3.5-2B (local, `/home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B`)
**Dataset:** ImageNet-1K val (`/opt/ImageNet/val`)
**Hardware:** NVIDIA RTX 4090 D (48 GB), bf16

## Unit tests

All 21 unit tests pass (see `TEST_REPORT_unit.txt`).

| Test module | Cases | Result |
|---|---|---|
| `tests/test_config.py` | 3 | PASS |
| `tests/test_prompts.py` | 4 | PASS |
| `tests/test_postprocess.py` | 5 | PASS |
| `tests/test_metrics.py` | 4 | PASS |
| `tests/test_dataset.py` | 4 | PASS |
| `tests/test_inference.py` | 1 | PASS |

Runtime: 7.78 s.

## Smoke test (real model)

See `TEST_REPORT_smoke.txt`. Outcome: **PASS** in 234.45 s (about 4 min, dominated by first-time vLLM model load + torch.compile warm-up).

- Model architecture resolved to `Qwen3_5ForConditionalGeneration`
- vLLM 0.19.0 engine initialized with max_model_len=8192, KV cache = 746,912 tokens
- Single-image prompt processed in 2.22 s at 1700 in-toks/s, 7.66 out-toks/s
- Returned dict shape matches spec `{pred_class, pred_idx, confidence, raw, parse_level}`

## 50-sample subset run

See `TEST_REPORT_subset.txt`.

- **Top-1 accuracy:** 96.00 %
- **parse_level_ratio:** `{"1": 1.00}` — every output was parsed as strict JSON (no fuzzy / token-scan fallback needed)
- **Throughput (fresh run):** 3.53 samples/sec
- **Average latency:** 283 ms/sample
- **Wall time:** 14.16 s (inference only, after model load)
- **Model load (warm):** 55–60 s; **cold:** 177 s

Note: the 50 samples come from the first alphabetically-sorted synset (`n01440764` = "tench"), so the 96 % accuracy is not representative of full-dataset performance — it confirms the pipeline works correctly. A full val-set run is needed for a representative top-1 number.

## Resume verification

See `TEST_REPORT_resume.txt`.

- Initial run: 50 predictions written, `top1=0.9600`
- Re-invocation with `--resume --num-samples 50 --run-id smoke_subset_50`:
  - `predictions.jsonl` line count: **50 → 50** (no duplicates)
  - `metrics.json` `total_samples`: 50
  - `metrics.json` `top1_acc`: still **0.9600** (preserved because metrics are recomputed from the complete `predictions.jsonl`)
  - `metrics.json` `throughput.new_samples_this_run`: 0

The initial implementation overwrote `metrics.json` with zeros when no new samples were processed — caught during verification, fixed in commit `74ca7d2` (metrics now recomputed from the persisted JSONL rather than the in-memory new-only buffer).

## Static checks

- `python -m compileall main.py dataset model prompts utils scripts/build_class_names.py`: **PASS** (no output)
- Import graph: `import main, model.inference, dataset.imagenet, dataset.imagenet_lt, utils.config, utils.metrics, utils.postprocess, prompts.templates` → **PASS**
- Shell scripts: `bash -n scripts/run_inference.sh scripts/run_finetune.sh` → **PASS**

## Artefacts on disk

```
outputs/smoke_subset_50/
├── predictions.jsonl        # 50 lines
├── metrics.json             # 96.00 % top-1
├── config.yaml              # config snapshot
├── run.log                  # main log
└── corrupt_images.log       # empty (no corrupt images in first synset)
```

## Known issues / follow-ups

1. The `per_class_acc` field only contains classes actually seen in the subset (by design). Full run will populate all 1000 entries.
2. Fine-tuning (`model/finetune.py`) raises `NotImplementedError` by design; Stage-2 scope.
3. `ImageNetLTDataset` is a pass-through stub; the real LT sampler is Stage-2 scope.
4. `top5_acc` is a placeholder (`null`); the current prompt only emits top-1. Adding top-k requires a prompt+parser change.
5. The subset top-1 of 96 % comes entirely from one synset (`n01440764`) because the subset slice is sequential. The next experiment should either shuffle with a fixed seed before sampling, or run over the full val set, to get a representative zero-shot accuracy number.

## Sign-off

All verification items in Section 8 of the spec (`docs/superpowers/specs/2026-04-21-qwen-vlm-classification-design.md`) have been executed with evidence recorded in this directory. Stage-1 implementation is complete and ready for Stage-2 extensions.
