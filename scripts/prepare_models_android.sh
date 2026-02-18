#!/bin/bash
# ============================================================
# Prepare models for Android APK deployment
#
# Steps:
#   1. Quantize MT ONNX models to INT8 (if not done)
#   2. Quantize TTS ONNX model to INT8 (if not done)
#   3. Copy all required models to Android assets/
#   4. Report total size
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODELS_DIR="$PROJECT_ROOT/models"
ASSETS_DIR="$PROJECT_ROOT/android/app/src/main/assets/models"

MT_DIR="$MODELS_DIR/mt/onnx"
TTS_DIR="$MODELS_DIR/tts"
ASR_MODEL="$MODELS_DIR/asr/ggml-base.en.bin"

echo "=== Preparing Models for Android ==="
echo "Source: $MODELS_DIR"
echo "Target: $ASSETS_DIR"
echo ""

# -------------------------------------------------------
# Step 1: Quantize MT models to INT8
# -------------------------------------------------------
echo "--- Step 1: Quantize MT models (INT8) ---"

NEED_QUANTIZE=false
for name in encoder_model decoder_model decoder_with_past_model; do
    if [ -f "$MT_DIR/${name}.onnx" ] && [ ! -f "$MT_DIR/${name}_int8.onnx" ]; then
        NEED_QUANTIZE=true
        break
    fi
done

if [ "$NEED_QUANTIZE" = true ]; then
    echo "Running quantization..."
    python3 "$SCRIPT_DIR/quantize_models.py"
else
    echo "INT8 models already exist — skipping quantization"
fi

# Verify INT8 models
MT_OK=true
for name in encoder_model decoder_model decoder_with_past_model; do
    if [ -f "$MT_DIR/${name}_int8.onnx" ]; then
        SIZE=$(du -h "$MT_DIR/${name}_int8.onnx" | cut -f1)
        echo "  OK: ${name}_int8.onnx ($SIZE)"
    elif [ -f "$MT_DIR/${name}.onnx" ]; then
        echo "  WARN: ${name}_int8.onnx missing, will use FP32"
        MT_OK=false
    else
        echo "  SKIP: ${name}.onnx not found"
    fi
done

# -------------------------------------------------------
# Step 2: Quantize TTS model to INT8
# -------------------------------------------------------
echo ""
echo "--- Step 2: Quantize TTS model (INT8) ---"

if [ -f "$TTS_DIR/hi_IN-rohan-medium.onnx" ] && [ ! -f "$TTS_DIR/hi_IN-rohan-medium_int8.onnx" ]; then
    echo "Running TTS quantization..."
    python3 -c "
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
src = '$TTS_DIR/hi_IN-rohan-medium.onnx'
dst = '$TTS_DIR/hi_IN-rohan-medium_int8.onnx'
quantize_dynamic(src, dst, weight_type=QuantType.QInt8)
in_sz = os.path.getsize(src) / (1024*1024)
out_sz = os.path.getsize(dst) / (1024*1024)
print(f'  TTS: {in_sz:.0f} MB → {out_sz:.0f} MB ({out_sz/in_sz*100:.0f}%)')
"
elif [ -f "$TTS_DIR/hi_IN-rohan-medium_int8.onnx" ]; then
    SIZE=$(du -h "$TTS_DIR/hi_IN-rohan-medium_int8.onnx" | cut -f1)
    echo "  OK: hi_IN-rohan-medium_int8.onnx ($SIZE)"
fi

# -------------------------------------------------------
# Step 3: Copy to Android assets
# -------------------------------------------------------
echo ""
echo "--- Step 3: Copy models to Android assets ---"

# Clean previous assets
rm -rf "$ASSETS_DIR"
mkdir -p "$ASSETS_DIR/mt/onnx" "$ASSETS_DIR/tts"

# ASR model
if [ -f "$ASR_MODEL" ]; then
    cp "$ASR_MODEL" "$ASSETS_DIR/ggml-base.en.bin"
    echo "  Copied: ggml-base.en.bin"
else
    echo "  ERROR: ASR model not found at $ASR_MODEL"
fi

# MT models (prefer INT8, fall back to FP32)
for name in encoder_model decoder_model decoder_with_past_model; do
    if [ -f "$MT_DIR/${name}_int8.onnx" ]; then
        # Copy INT8 model with original name (so code doesn't need path changes)
        cp "$MT_DIR/${name}_int8.onnx" "$ASSETS_DIR/mt/onnx/${name}.onnx"
        echo "  Copied: mt/onnx/${name}.onnx (INT8)"
    elif [ -f "$MT_DIR/${name}.onnx" ]; then
        cp "$MT_DIR/${name}.onnx" "$ASSETS_DIR/mt/onnx/${name}.onnx"
        echo "  Copied: mt/onnx/${name}.onnx (FP32 — WARNING: large)"
    fi
done

# MT tokenizer files
for f in source.spm target.spm vocab.json config.json generation_config.json \
         tokenizer_config.json special_tokens_map.json; do
    if [ -f "$MT_DIR/$f" ]; then
        cp "$MT_DIR/$f" "$ASSETS_DIR/mt/onnx/$f"
        echo "  Copied: mt/onnx/$f"
    fi
done

# TTS model (prefer INT8)
if [ -f "$TTS_DIR/hi_IN-rohan-medium_int8.onnx" ]; then
    cp "$TTS_DIR/hi_IN-rohan-medium_int8.onnx" "$ASSETS_DIR/tts/hi_IN-rohan-medium.onnx"
    echo "  Copied: tts/hi_IN-rohan-medium.onnx (INT8)"
elif [ -f "$TTS_DIR/hi_IN-rohan-medium.onnx" ]; then
    cp "$TTS_DIR/hi_IN-rohan-medium.onnx" "$ASSETS_DIR/tts/hi_IN-rohan-medium.onnx"
    echo "  Copied: tts/hi_IN-rohan-medium.onnx (FP32)"
fi

# TTS config
if [ -f "$TTS_DIR/hi_IN-rohan-medium.onnx.json" ]; then
    cp "$TTS_DIR/hi_IN-rohan-medium.onnx.json" "$ASSETS_DIR/tts/hi_IN-rohan-medium.onnx.json"
    echo "  Copied: tts/hi_IN-rohan-medium.onnx.json"
fi

# -------------------------------------------------------
# Step 3b: LLM model (Qwen3-0.6B .pte)
# -------------------------------------------------------
echo ""
echo "--- Step 3b: LLM model (Qwen3-0.6B) ---"

LLM_PTE="$MODELS_DIR/llm/qwen3_0.6B.pte"
LLM_TOK="$MODELS_DIR/llm/Qwen3-0.6B/tokenizer.json"
LLM_DEST="$ASSETS_DIR/llm"

if [ -f "$LLM_PTE" ]; then
    mkdir -p "$LLM_DEST"
    cp "$LLM_PTE" "$LLM_DEST/"
    echo "  Copied: llm/qwen3_0.6B.pte ($(du -h "$LLM_PTE" | cut -f1))"
    if [ -f "$LLM_TOK" ]; then
        cp "$LLM_TOK" "$LLM_DEST/"
        echo "  Copied: llm/tokenizer.json"
    fi
else
    echo "  SKIP: LLM model not found at $LLM_PTE"
    echo "  Run ./scripts/export_llm_model.sh to create it"
fi

# -------------------------------------------------------
# Step 4: Size report
# -------------------------------------------------------
echo ""
echo "--- Step 4: Size Report ---"
echo ""

TOTAL=0
for f in $(find "$ASSETS_DIR" -type f); do
    SIZE_BYTES=$(stat --printf="%s" "$f" 2>/dev/null || stat -f "%z" "$f" 2>/dev/null || echo 0)
    SIZE_MB=$(echo "scale=1; $SIZE_BYTES / 1048576" | bc 2>/dev/null || echo "?")
    REL=$(echo "$f" | sed "s|$ASSETS_DIR/||")
    printf "  %-50s %s MB\n" "$REL" "$SIZE_MB"
    TOTAL=$((TOTAL + SIZE_BYTES))
done

TOTAL_MB=$(echo "scale=1; $TOTAL / 1048576" | bc 2>/dev/null || echo "?")
echo ""
echo "  TOTAL: ${TOTAL_MB} MB"

if [ "$TOTAL" -gt 419430400 ]; then  # 400 MB
    echo ""
    echo "  WARNING: Total exceeds 400 MB mobile budget!"
    echo "  Consider removing decoder_with_past_model.onnx"
    echo "  or further quantizing models."
else
    echo "  OK: Under 400 MB mobile budget."
fi

echo ""
echo "=== Model preparation complete ==="
echo "Assets directory: $ASSETS_DIR"
echo ""
echo "Next: cd android && ./gradlew assembleDebug"
