#!/usr/bin/env bash
# Exact one-GPU, one-epoch recipe used for the final all-public model.

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 BASE_MODEL TRAIN_DATA_JSONL OUTPUT_DIR" >&2
  exit 2
fi

BASE_MODEL=$(realpath "$1")
TRAIN_DATA=$(realpath "$2")
OUTPUT_DIR=$(realpath -m "$3")

if [[ ! -s "$BASE_MODEL/config.json" ]]; then
  echo "Qwen3.8-27B config not found: $BASE_MODEL/config.json" >&2
  exit 1
fi
if [[ ! -s "$TRAIN_DATA" ]]; then
  echo "Prepared MS-SWIFT JSONL not found: $TRAIN_DATA" >&2
  exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "Refusing to reuse output path: $OUTPUT_DIR" >&2
  exit 1
fi
if ! command -v swift >/dev/null 2>&1; then
  echo "MS-SWIFT is not installed or its CLI is not on PATH" >&2
  exit 1
fi

python - "$BASE_MODEL/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    config = json.load(handle)
if config.get("model_type") != "qwen3_5":
    raise SystemExit("BASE_MODEL is not from the expected Qwen3.8 multimodal family")
if config.get("language_model_only") is not False:
    raise SystemExit("BASE_MODEL must contain the vision encoder")
text = config.get("text_config", {})
if text.get("num_hidden_layers") != 64 or text.get("hidden_size") != 5120:
    raise SystemExit("BASE_MODEL is not the dense Qwen3.8-27B layout")
if config.get("quantization_config") is not None:
    raise SystemExit("Train from the unquantized BF16 base; export FP8 after merging")
PY

python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

required = {"ms-swift": "4.4.2", "transformers": "5.8.0", "peft": "0.19.1"}
for package, minimum in required.items():
    installed = Version(version(package))
    print(f"{package}={installed}")
    if installed < Version(minimum):
        raise SystemExit(f"{package}>={minimum} is required")
PY

python - <<'PY'
import torch

if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("This reproduction script expects exactly one visible CUDA GPU")
props = torch.cuda.get_device_properties(0)
memory_mib = props.total_memory // 1024**2
print(f"gpu={props.name} memory_mib={memory_mib} capability={props.major}.{props.minor}")
if props.major < 8 or memory_mib < 75000:
    raise SystemExit("The exact recipe requires a BF16-capable GPU with about 80 GB VRAM")
PY

export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export IMAGE_MAX_TOKEN_NUM=1024
export TOKENIZERS_PARALLELISM=false

swift sft \
  --model "$BASE_MODEL" \
  --model_type qwen3_5 \
  --add_version true \
  --dataset "$TRAIN_DATA" \
  --enable_thinking false \
  --add_non_thinking_prefix true \
  --loss_scale ignore_empty_think \
  --torch_dtype bfloat16 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --max_length 2048 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --lr_scheduler_type cosine \
  --max_grad_norm 1.0 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --report_to tensorboard \
  --output_dir "$OUTPUT_DIR" \
  --seed 20260728 \
  --data_seed 20260728 \
  --tuner_type lora \
  --freeze_llm false \
  --freeze_vit false \
  --freeze_aligner true \
  --target_modules all-linear \
  --modules_to_save model.visual.merger \
  --learning_rate 5e-5 \
  --vit_lr 1e-5 \
  --aligner_lr 1e-5 \
  --lora_rank 32 \
  --lora_alpha 64 \
  --lora_dropout 0.05 \
  --gradient_checkpointing true \
  --num_train_epochs 1 \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps 1000 \
  --save_total_limit 2 \
  --save_only_model true \
  --logging_steps 25 \
  --disable_tqdm true \
  --load_from_cache_file true

