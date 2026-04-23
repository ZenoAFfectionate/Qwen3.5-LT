# Qwen3.5-VL 图像分类代码框架设计

bash scripts/run_inference.sh --resume --full --run-id full_val_20260421

- **Date**: 2026-04-21
- **Project**: `/home/kemove/Experiment/Qwen3.5-LT`
- **Goal**: 基于 vLLM 引擎，用 Qwen3.5-2B 多模态模型在 ImageNet-1K 平衡数据集上完成 zero-shot 图像分类；代码框架预留长尾（LT）扩展与微调接口。

## 1. 范围

**第一阶段（本 spec 覆盖）**：
- 在 ImageNet-1K 验证集（或子集）上完成 zero-shot 分类评测
- 输出 JSON 格式预测 + Top-1/Top-5 指标 + 解析质量统计
- 留出长尾数据集与微调的接口占位，不实现内部逻辑

**非目标**：
- 不实现 LoRA/QLoRA/全参微调的训练流程（仅占位接口）
- 不实现 Many/Medium/Few 长尾切分评测的实际逻辑（字段预留）
- 不实现 vLLM 在线服务模式

## 2. 技术选型

| 维度 | 选择 | 备选 / 理由 |
|---|---|---|
| 推理引擎 | vLLM 离线批推理（`vllm.LLM` 类） | vs. 在线 serve：MVP 阶段离线批处理最简 |
| 模型 | `/home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B` | 本地 HF 标准目录 |
| 数据集 | ImageNet-1K（`/opt/ImageNet/`，ImageFolder 结构） | 验证集 50K 张，按 synset 子目录组织 |
| 输出格式 | JSON：`{"class": str, "confidence": float}` | vs. 自由文本 / logit 打分：JSON 便于统计 |
| 后处理 | 三级兜底：JSON 解析 → 模糊匹配 → token 匹配 | 保证任何输出都能得到一个预测 |
| 配置管理 | YAML + dataclass（`configs/*.yaml`） | vs. 纯 CLI：便于 LT 扩展时复用代码 |
| 微调 | 本阶段占位接口（`NotImplementedError`） | 下一阶段再决定 LoRA / QLoRA |

## 3. 目录结构

```
Qwen3.5-LT/
├── configs/
│   ├── base.yaml              # 公共字段
│   ├── imagenet.yaml          # 平衡数据集
│   └── imagenet_lt.yaml       # 预留长尾配置
├── dataset/
│   ├── __init__.py
│   ├── base.py                # ClassificationDataset 抽象基类
│   ├── imagenet.py            # ImageNet ImageFolder 加载
│   └── imagenet_lt.py         # 预留占位（继承 ImageNet）
├── model/
│   ├── __init__.py
│   ├── inference.py           # VLMClassifier（vLLM 推理 + JSON 解析）
│   └── finetune.py            # 占位 FinetuneRunner
├── prompts/
│   ├── __init__.py
│   ├── templates.py           # SYSTEM_PROMPT / USER_PROMPT_TEMPLATE
│   └── imagenet_classes.txt   # 1000 行，每行一个类名
├── utils/
│   ├── __init__.py
│   ├── config.py              # yaml → dataclass
│   ├── logger.py              # 统一 logging
│   ├── metrics.py             # Top-1/Top-5/Per-class
│   └── postprocess.py         # 三级 JSON 解析
├── scripts/
│   ├── run_inference.sh       # 推理脚本（参数化：支持全量 / 子集 / 自定义配置）
│   └── run_finetune.sh        # 占位
├── tests/                     # 测试（见第 8 节）
│   ├── test_dataset.py
│   ├── test_postprocess.py
│   ├── test_prompt.py
│   ├── test_config.py
│   └── test_smoke.py
├── main.py                    # 入口
├── requirements.txt
└── README.md
```

## 4. 核心接口

### 4.1 数据集

```python
# dataset/base.py
class ClassificationDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx) -> tuple[PIL.Image, int, str]:
        """returns (image, label_idx, synset_or_key)"""
    @property
    def class_names(self) -> list[str]: ...
    @property
    def synset_to_idx(self) -> dict[str, int]: ...
```

### 4.2 推理

```python
# model/inference.py
class VLMClassifier:
    def __init__(self, cfg): ...
    def classify_batch(
        self,
        images: list[PIL.Image],
        class_names: list[str],
    ) -> list[dict]:
        """return: [{"pred_class", "pred_idx", "confidence", "raw", "parse_level"}, ...]"""
```

### 4.3 微调（占位）

```python
# model/finetune.py
class FinetuneRunner:
    def __init__(self, cfg): ...
    def train(self):
        raise NotImplementedError("Stage 2: LoRA fine-tuning")
```

## 5. Prompt 设计

```python
SYSTEM_PROMPT = """You are an expert image classifier. Given an image, \
identify the most likely category from the provided candidate list. \
Respond with ONLY a valid JSON object, no extra text, in the form:
{"class": "<exact category name from the list>", "confidence": <float 0-1>}"""

USER_PROMPT_TEMPLATE = """Classify this image into exactly one of the following {n} ImageNet categories:

{class_list}

Return only the JSON object."""
```

- 默认注入全部 1000 类（闭集选择，减少幻觉）
- 开关 `prompt.inject_all_classes: false` 切换为自由生成 + 模糊匹配

## 6. JSON 三级兜底解析

```python
def parse_prediction(raw: str, class_names: list[str]) -> dict:
    # 级别 1: 正则抽取第一个 {...} → json.loads
    # 级别 2: JSON 解析成功但 class 不在列表 → rapidfuzz 最近邻
    # 级别 3: JSON 解析失败 → 对 raw 全文做 token 匹配
    # 全部失败: pred_idx=-1, confidence=0.0
    return {"pred_class", "pred_idx", "confidence", "raw", "parse_level"}
```

`parse_level` 字段写入 `predictions.jsonl`，用于事后 prompt 质量分析。

## 7. 配置文件（示例）

```yaml
# configs/imagenet.yaml
model:
  path: /home/kemove/.cache/huggingface/hub/models--Qwen--Qwen3.5-2B
  trust_remote_code: true
  dtype: bfloat16
  gpu_memory_utilization: 0.85
  max_model_len: 4096
  limit_mm_per_prompt: {image: 1}

dataset:
  name: imagenet
  root: /opt/ImageNet
  split: val
  classes_file: prompts/imagenet_classes.txt

inference:
  batch_size: 16
  temperature: 0.0
  max_tokens: 64

prompt:
  template: classify_closed_set
  inject_all_classes: true

eval:
  num_samples: null         # null=全量；--num-samples N 覆盖
  save_predictions: true
  output_dir: outputs/

run_id: null                # null 自动生成
```

## 8. 测试与验证

本节明确实现完成后必须通过的验证项。**每一个测试都必须真实运行并产出日志，不接受"看起来对"的声明**。

### 8.1 单元测试（pytest，`tests/`）

| 文件 | 覆盖内容 | 验收 |
|---|---|---|
| `test_config.py` | YAML 加载、字段校验、CLI 覆盖 | `pytest tests/test_config.py` 全绿 |
| `test_postprocess.py` | 三级解析：合法 JSON / 带噪 JSON / 错误类名 / 非 JSON 文本 | 每级至少 3 个 case |
| `test_prompt.py` | prompt 生成、类名注入长度、占位符替换 | 生成字符串包含所有类名 |
| `test_dataset.py` | ImageFolder 路径、class_names 顺序、`__getitem__` 返回类型 | 用 2 个 mock 类 × 3 张图的小夹具 |

### 8.2 冒烟测试（`tests/test_smoke.py`，可选 GPU 标记）

- **目的**：真实加载模型并对一张图完成一次端到端推理
- **内容**：
  1. 加载配置
  2. `VLMClassifier` 初始化
  3. 对 `/opt/ImageNet/val/n01440764/` 下首张图做 `classify_batch(batch_size=1)`
  4. 断言：返回 dict 含预期键、`pred_idx ∈ [0, 999]`、`raw` 非空
- **跳过条件**：无 GPU 或模型不可达时通过 `pytest.mark.skipif` 跳过，并打印原因

### 8.3 子集评测验证

- 命令：`bash scripts/run_inference.sh --num-samples 50`
- 验收：
  - 脚本非零退出码时直接失败
  - `outputs/<run_id>/predictions.jsonl` 行数 = 50
  - `outputs/<run_id>/metrics.json` 存在且包含 `top1_acc`、`parse_level_ratio`、`throughput`
  - `parse_level_ratio["1"]` 应 ≥ 0.5（prompt 工程合理下限）
  - Top-1 acc 应 > 0（即便模型弱也不可能零正确）
  - `throughput.samples_per_sec` > 0 且 `wall_time_sec` 与墙钟时间吻合

### 8.4 鲁棒性验证

- **损坏图像**：测试集中塞一张 0 字节的 JPEG，断言写入 `corrupt_images.log` 且主流程不崩
- **断点续跑**：运行 10 张后 `Ctrl-C`，带 `--resume` 重启，断言 `predictions.jsonl` 最终行数 = 期望 subset 大小
- **输出格式**：`predictions.jsonl` 每行能被 `json.loads` 成功

### 8.5 静态检查

- 所有 Python 文件通过 `python -m py_compile`（语法正确）
- `python -c "import main, model.inference, dataset.imagenet, utils.config"` 无 ImportError

### 8.6 验证报告

实现结束后，在 PR / 交付说明中必须粘贴：
- `pytest -v` 完整输出
- `python main.py --config configs/imagenet.yaml --num-samples 50` 的完整 stdout + stderr
- `outputs/<run_id>/metrics.json` 内容
- 各级测试通过/跳过/失败的明确声明

**完成标准**：以上 5 项全部通过（或跳过有明确理由），才能把对应任务标记 completed。

## 9. 错误处理

| 场景 | 处理 |
|---|---|
| 模型加载失败 | fail fast，抛异常 |
| 图像损坏（`PIL.UnidentifiedImageError`、空文件） | 跳过 + 记 `corrupt_images.log` |
| JSON 解析全失败 | `pred_idx=-1`，计入 Top-1 错误，写 `failed_parses.jsonl` |
| 推理中断 | 流式 append 到 `predictions.jsonl`；`--resume` 从已有行数继续 |

## 10. 输出产物

```
outputs/<run_id>/
├── predictions.jsonl    # 每行: {idx, synset, gt_class, pred_class, pred_idx, confidence, parse_level, raw}
├── metrics.json         # 汇总指标（含吞吐）
├── config.yaml          # 本次运行的配置快照
├── corrupt_images.log
├── failed_parses.jsonl
└── run.log
```

`metrics.json` 除分类指标外，还记录推理效率：
```json
{
  "top1_acc": 0.xx,
  "top5_acc": 0.xx,
  "per_class_acc": {...},
  "parse_level_ratio": {"1": 0.x, "2": 0.x, "3": 0.x, "fail": 0.x},
  "throughput": {
    "total_samples": 50000,
    "wall_time_sec": 1234.5,
    "samples_per_sec": 40.5,
    "avg_latency_ms_per_sample": 24.7,
    "model_load_sec": 45.2
  }
}
```

## 11. 运行脚本（`scripts/`）

**11.1 `scripts/run_inference.sh`**：统一推理脚本，用位置/命名参数切换模式

```bash
#!/usr/bin/env bash
# 用途: 运行 Qwen3.5-2B 在 ImageNet 上的推理评测
# 用法:
#   bash scripts/run_inference.sh                       # 默认: 全量 val
#   bash scripts/run_inference.sh --num-samples 200     # 子集模式
#   bash scripts/run_inference.sh --config configs/imagenet_lt.yaml --full
#   bash scripts/run_inference.sh --resume --run-id xxx # 断点续跑
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/imagenet.yaml"
EXTRA_ARGS=()
RUN_ID=""

# 解析参数
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2"; shift 2 ;;
        --num-samples)   EXTRA_ARGS+=("--num-samples" "$2"); shift 2 ;;
        --full)          EXTRA_ARGS+=("--full"); shift ;;
        --resume)        EXTRA_ARGS+=("--resume"); shift ;;
        --run-id)        RUN_ID="$2"; shift 2 ;;
        *)               EXTRA_ARGS+=("$1"); shift ;;
    esac
done

# 自动生成 run_id（未显式指定时）
if [[ -z "${RUN_ID}" ]]; then
    CFG_NAME="$(basename "${CONFIG}" .yaml)"
    RUN_ID="${CFG_NAME}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p outputs
python main.py \
    --config "${CONFIG}" \
    --mode eval \
    --run-id "${RUN_ID}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "outputs/${RUN_ID}.log"
```

参数覆盖能力：
- `--config <path>`：切换数据集配置（`imagenet.yaml` / `imagenet_lt.yaml` / 自定义）
- `--num-samples N`：zero-shot 子集评测
- `--full`：zero-shot 全量评测（默认行为）
- `--resume`：从已有 `predictions.jsonl` 行数续跑
- `--run-id`：自定义输出目录名

Zero-shot 模式切换完全由 `--config` + 上述参数完成，无需额外脚本。

**11.2 `scripts/run_finetune.sh`**（占位）：当前调用 `main.py --mode finetune`，内部抛 NotImplementedError 并提示 "Stage 2"。

**脚本验收**：
- 使用 `set -euo pipefail`，任何异常立即退出
- `chmod +x scripts/*.sh`
- `bash scripts/run_inference.sh --num-samples 50` 必须在测试阶段真实跑通

## 12. main.py CLI

```bash
python main.py --config configs/imagenet.yaml \
               --mode eval \
               [--num-samples 100] \
               [--full] \
               [--run-id my_run_01] \
               [--resume]
```

- `--mode`: `eval` | `finetune`（后者仅调用占位接口，抛 NotImplementedError）
- `--num-samples` 与 `--full` 互斥，覆盖 `eval.num_samples`

## 13. 未来扩展（不在本 spec 实现）

- **长尾数据集**：`dataset/imagenet_lt.py` 基于 ImageNet 样本 + 长尾索引文件做重采样
- **Many/Medium/Few 评测**：`utils/metrics.py` 接入 split 映射
- **LoRA 微调**：`model/finetune.py` 集成 PEFT + trl/transformers Trainer
- **In-context few-shot**：prompt 模板里注入同类样本作为引导
- **vLLM 在线服务**：切换到 `vllm serve` + OpenAI 客户端模式

## 14. 依赖清单（`requirements.txt`）

```
vllm>=0.6.0
torch
torchvision
transformers
Pillow
pyyaml
rapidfuzz
tqdm
pytest
```
