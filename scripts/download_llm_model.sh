#!/usr/bin/env bash
# Download Qwen3-0.6B model files for LLM translation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$PROJECT_ROOT/models/llm"

mkdir -p "$MODEL_DIR"

# Download Qwen3-0.6B-Instruct from HuggingFace
# The actual .pte file must be exported using export_llm_model.sh
# This script downloads the HuggingFace weights needed for export

if [ -d "$MODEL_DIR/Qwen3-0.6B" ]; then
    echo "Qwen3-0.6B already downloaded in $MODEL_DIR/Qwen3-0.6B"
    exit 0
fi

echo "=== Downloading Qwen3-0.6B-Instruct ==="
echo "This requires the 'huggingface_hub' Python package."
echo "Install with: pip install huggingface_hub"

python3 -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='Qwen/Qwen3-0.6B',
    local_dir='$MODEL_DIR/Qwen3-0.6B',
    allow_patterns=['*.json', '*.safetensors', '*.model', '*.tiktoken', '*.txt'],
)
print('Download complete: $MODEL_DIR/Qwen3-0.6B')
"

echo "=== Done ==="
echo "Next step: run ./scripts/export_llm_model.sh to export to .pte format"
