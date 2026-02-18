#!/usr/bin/env bash
# Export Qwen3-0.6B to ExecuTorch .pte format with INT4 quantization
# Requires: pip install executorch torch torchao
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models/llm"
OUTPUT_PTE="$MODEL_DIR/qwen3_0.6B.pte"

if [ -f "$OUTPUT_PTE" ]; then
    echo "Model already exported: $OUTPUT_PTE"
    exit 0
fi

if [ ! -d "$MODEL_DIR/Qwen3-0.6B" ]; then
    echo "ERROR: Qwen3-0.6B not downloaded. Run ./scripts/download_llm_model.sh first."
    exit 1
fi

echo "=== Exporting Qwen3-0.6B to ExecuTorch ==="

# Step 1: Convert HuggingFace weights to ExecuTorch format
echo "Step 1: Converting weights..."
python3 -m executorch.examples.models.qwen3.convert_weights \
    "$MODEL_DIR/Qwen3-0.6B" \
    "$MODEL_DIR/qwen3_converted.bin"

# Step 2: Download config if not present
CONFIG_FILE="$MODEL_DIR/0_6b_config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Step 2: Downloading config..."
    curl -L -o "$CONFIG_FILE" \
        "https://raw.githubusercontent.com/pytorch/executorch/main/examples/models/qwen3/config/0_6b_config.json"
fi

# Step 3: Export with XNNPACK backend + KleidiAI + INT4 quantization
echo "Step 3: Exporting to .pte with 8da4w quantization..."
python3 -m executorch.extension.llm.export.export_llm \
    --model "qwen3_0_6b" \
    --checkpoint "$MODEL_DIR/qwen3_converted.bin" \
    --params "$CONFIG_FILE" \
    --output_name "$OUTPUT_PTE" \
    -kv --use_sdpa_with_kv_cache \
    -X --xnnpack-extended-ops \
    -qmode 8da4w --group_size 128 \
    --embedding-quantize 4,32 \
    --max_context_length 512 \
    --max_seq_length 128 \
    -d fp32

echo "=== Export complete ==="
echo "Model: $OUTPUT_PTE"
echo "Size: $(du -h "$OUTPUT_PTE" | cut -f1)"
echo ""
echo "To use: ./build-host/translation_host --llm-model $OUTPUT_PTE --llm-tokenizer $MODEL_DIR/Qwen3-0.6B/tokenizer.json ..."
