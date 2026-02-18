#!/usr/bin/env bash
# Download OPUS-MT EN→HI model and export to ONNX.
# Requires: pip install transformers optimum[exporters] sentencepiece
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../models/mt/onnx"

mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/encoder_model.onnx" ]; then
    echo "MT model already exists in ${MODEL_DIR}"
    exit 0
fi

echo "Exporting Helsinki-NLP/opus-mt-en-hi to ONNX..."
python3 -m optimum.exporters.onnx \
    --model Helsinki-NLP/opus-mt-en-hi \
    --task text2text-generation-with-past \
    "${MODEL_DIR}"

echo "Copying tokenizer files..."
python3 -c "
from transformers import MarianTokenizer
tok = MarianTokenizer.from_pretrained('Helsinki-NLP/opus-mt-en-hi')
tok.save_pretrained('${MODEL_DIR}')
"

echo "MT model exported to: ${MODEL_DIR}"
