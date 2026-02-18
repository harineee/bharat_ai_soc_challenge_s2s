#!/bin/bash
# Build the host (desktop) translation pipeline.
# Requires: cmake, g++/clang++, ONNX Runtime, SentencePiece
# Optional: PortAudio (for real-time microphone mode)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_ROOT/build-host"

echo "=== Building Host Translation Pipeline ==="
echo "Project root: $PROJECT_ROOT"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake "$PROJECT_ROOT/host" \
    -DCMAKE_BUILD_TYPE=Release

cmake --build . -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)"

echo ""
echo "Build complete: $BUILD_DIR/translation_host"
echo ""
echo "Usage:"
echo "  $BUILD_DIR/translation_host \\"
echo "    --asr-model models/asr/ggml-base.en.bin \\"
echo "    --mt-model models/mt/onnx/encoder_model.onnx \\"
echo "    --tts-model models/tts/hi_IN-rohan-medium.onnx \\"
echo "    --input test.wav --output hindi.wav"
