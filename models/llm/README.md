# LLM Models

This directory holds the Qwen3-0.6B model files for LLM-based translation.

## Files (after download and export)

- `Qwen3-0.6B/` — HuggingFace model files (weights, tokenizer, config)
- `qwen3_0.6B.pte` — ExecuTorch exported model (~472 MB, INT4 quantized)

## Quantization

- **Scheme**: 8da4w (8-bit dynamic activation, 4-bit weights)
- **Group size**: 128
- **Embedding quantization**: 4-bit
- **Framework**: ExecuTorch with XNNPACK + KleidiAI backend

## Setup

1. Download: `./scripts/download_llm_model.sh`
2. Export: `./scripts/export_llm_model.sh`

## Usage in Translation Modes

- **SPEED mode**: LLM not used (NMT only)
- **BALANCED mode**: LLM verifies NMT draft via speculative decoding (~150ms)
- **QUALITY mode**: LLM generates translation autoregressively (~500ms)
