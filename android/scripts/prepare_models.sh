#!/bin/bash
# Copy models from models/ into Android APK assets.
# Run AFTER quantization (scripts/quantize_models.py) for smaller APK.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
ASSETS="$SCRIPT_DIR/../app/src/main/assets/models"

echo "=== Preparing Android Model Assets ==="
echo "Source: $PROJECT_ROOT/models"
echo "Target: $ASSETS"

mkdir -p "$ASSETS/mt/onnx" "$ASSETS/tts"

# ASR model 
cp "$PROJECT_ROOT/models/asr/ggml-base.en.bin" "$ASSETS/" 2>/dev/null || echo "WARN: ASR model not found"

# MT models (use INT8 if available, else FP32)
for name in encoder_model decoder_model decoder_with_past_model; do
    if [ -f "$PROJECT_ROOT/models/mt/onnx/${name}_int8.onnx" ]; then
        cp "$PROJECT_ROOT/models/mt/onnx/${name}_int8.onnx" "$ASSETS/mt/onnx/${name}.onnx"
        echo "  MT: ${name}_int8.onnx → ${name}.onnx"
    elif [ -f "$PROJECT_ROOT/models/mt/onnx/${name}.onnx" ]; then
        cp "$PROJECT_ROOT/models/mt/onnx/${name}.onnx" "$ASSETS/mt/onnx/"
        echo "  MT: ${name}.onnx (FP32 — consider quantizing)"
    fi
done

# MT tokenizer files
for f in source.spm target.spm vocab.json config.json generation_config.json tokenizer_config.json special_tokens_map.json; do
    [ -f "$PROJECT_ROOT/models/mt/onnx/$f" ] && cp "$PROJECT_ROOT/models/mt/onnx/$f" "$ASSETS/mt/onnx/"
done

# TTS model
if [ -f "$PROJECT_ROOT/models/tts/hi_IN-rohan-medium_int8.onnx" ]; then
    cp "$PROJECT_ROOT/models/tts/hi_IN-rohan-medium_int8.onnx" "$ASSETS/tts/hi_IN-rohan-medium.onnx"
    echo "  TTS: hi_IN-rohan-medium_int8.onnx → hi_IN-rohan-medium.onnx"
elif [ -f "$PROJECT_ROOT/models/tts/hi_IN-rohan-medium.onnx" ]; then
    cp "$PROJECT_ROOT/models/tts/hi_IN-rohan-medium.onnx" "$ASSETS/tts/"
fi
cp "$PROJECT_ROOT/models/tts/hi_IN-rohan-medium.onnx.json" "$ASSETS/tts/" 2>/dev/null || true

echo ""
echo "Assets prepared. Total size:"
du -sh "$ASSETS"
echo ""
echo "WARNING: APK with all models will be large (~300+ MB)."
echo "Consider downloading models on first launch instead."
