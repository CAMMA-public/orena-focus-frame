#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MERGED_BF16_MODEL OFFICIAL_FP8_REFERENCE OUTPUT_DIR" >&2
  exit 2
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MODEL_DIR=$(realpath "$1")
REFERENCE_MODEL_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath -m "$3")

if [[ -e "$OUTPUT_DIR" || -e "${OUTPUT_DIR}.incomplete" ]]; then
  echo "Refusing to overwrite output or staging path: $OUTPUT_DIR" >&2
  exit 1
fi
for required in \
  "$MODEL_DIR/config.json" \
  "$MODEL_DIR/model.safetensors.index.json" \
  "$REFERENCE_MODEL_DIR/config.json" \
  "$REFERENCE_MODEL_DIR/model.safetensors.index.json"
do
  if [[ ! -s "$required" ]]; then
    echo "Required input is missing: $required" >&2
    exit 1
  fi
done

python - <<'PY'
import torch
import safetensors

print(f"torch={torch.__version__}")
print(f"safetensors={safetensors.__version__}")
if not torch.cuda.is_available() or not hasattr(torch, "float8_e4m3fn"):
    raise SystemExit("CUDA and PyTorch float8_e4m3fn support are required")
PY

python -u "$SCRIPT_DIR/export_qwen38_native_fp8.py" \
  --model-dir "$MODEL_DIR" \
  --reference-model-dir "$REFERENCE_MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0
