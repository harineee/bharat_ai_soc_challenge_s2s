#!/bin/bash
# Download Hindi VITS TTS model

set -e

MODEL_DIR="models"
TTS_MODEL_DIR="${MODEL_DIR}/tts"

echo "Downloading TTS model (Hindi VITS)"

mkdir -p "${TTS_MODEL_DIR}"

# Note: Indic-TTS models are PyTorch checkpoints
# Need to convert to ONNX for Android inference

echo "Hindi VITS model from Indic-TTS"
echo "Please download manually from:"
echo "  https://github.com/AI4Bharat/Indic-TTS"
echo ""
echo "Or use Python script:"
echo "  python scripts/convert_tts_to_onnx.py"

# Placeholder for actual download
echo "TTS model download script - manual conversion required"
echo "See docs/BUILD.md for conversion instructions"
