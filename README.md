# Qwen3.5-LT

Zero-shot image classification with **Qwen3.5-2B** (multimodal) via **vLLM** on
**ImageNet-1K**, with modular code ready to extend to long-tail benchmarks and
LoRA fine-tuning.

## Quick start

### Install

Dependencies (already satisfied in the `agent` conda env):

```bash
pip install -r requirements.txt
```

### One-time: build class-name list

```bash
python scripts/build_class_names.py
```

Produces `prompts/imagenet_classes.txt` (1000 lines, canonical WordNet names
derived from `/opt/ImageNet/devkit/data/meta.mat`).

### Run zero-shot evaluation

```bash
# quick subset (default: configs/imagenet.yaml, 200 samples if unspecified)
bash scripts/run_inference.sh --num-samples 200

# full validation set (50k images)
bash scripts/run_inference.sh --full

# long-tail placeholder config
bash scripts/run_inference.sh --config configs/imagenet_lt.yaml --num-samples 200

# resume an interrupted run (reuse the same --run-id)
bash scripts/run_inference.sh --resume --run-id <existing_run_id>
```

Outputs land in `outputs/<run_id>/`:
- `predictions.jsonl` – one prediction per line
- `metrics.json` – Top-1 / per-class / parse-level / throughput
- `config.yaml` – snapshot of the exact config used
- `run.log` – structured log
- `corrupt_images.log` / `failed_parses.jsonl` – diagnostics

### Fine-tune (Stage 2 placeholder)

```bash
bash scripts/run_finetune.sh
```

Currently raises `NotImplementedError`.

## Project layout

```
configs/      YAML configs (imagenet.yaml, imagenet_lt.yaml, base.yaml)
dataset/      ClassificationDataset base + ImageNetDataset + LT stub
model/        VLMClassifier (vLLM wrapper) + FinetuneRunner stub
prompts/      System/user prompt templates + 1000-line class list
utils/        Config loader, logger, metrics, 3-level JSON post-processor
scripts/      run_inference.sh, run_finetune.sh, build_class_names.py
tests/        pytest suite (unit + real-model smoke)
main.py       CLI entry
```

Full design: `docs/superpowers/specs/2026-04-21-qwen-vlm-classification-design.md`.
Implementation plan: `docs/superpowers/plans/2026-04-21-qwen-vlm-classification.md`.

## Tests

```bash
# fast unit tests (no GPU)
pytest -v --ignore=tests/test_smoke.py

# real-model smoke (loads Qwen3.5-2B via vLLM, ~3 min model load)
pytest -v tests/test_smoke.py

# skip smoke even when artifacts exist
QWEN35_SKIP_SMOKE=1 pytest -v tests/test_smoke.py
```

## Stage-1 measured performance

On 50 validation samples (first synset):
- **Top-1 accuracy**: 96.0 %
- **parse_level[1] (strict JSON)**: 100 %
- **Throughput**: 3.53 samples/sec (single RTX 4090 D, bf16)
- **Model load**: ~177 s cold, ~60 s warm (torch.compile cache hit)

See `docs/superpowers/plans/VERIFICATION_REPORT.md` for the full evidence trail.
