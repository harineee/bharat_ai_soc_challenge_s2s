# ARMM Project — Master Plan & Current Status

## What Is This Project

ARMM is an offline English→Hindi speech-to-speech translation app for Android phones. It's being built for the **Bharat AI-SoC Student Challenge** (deadline: **Feb 20, 2026**).

**The pipeline:**
```
English speech → [whisper.cpp ASR] → English text → [Translation] → Hindi text → [Piper TTS] → Hindi speech
```

**What makes our submission unique:**
We use **speculative decoding** — a technique where a fast NMT model (Marian, 20ms) generates a draft Hindi translation, then a Qwen3-0.6B LLM verifies and corrects the draft in a single forward pass (~80ms). This is 2-3x faster than naive LLM generation (~375ms) while maintaining LLM quality. Nobody else in the challenge will have this.

**Three adaptive modes:**
- **Speed**: NMT only (~120ms) — for when you need instant translation
- **Balanced**: NMT draft + LLM speculative verify (~200ms) — best tradeoff, default
- **Quality**: Full LLM autoregressive (~500ms) — highest accuracy

---

## Tech Stack

| Component | Technology | Format | Size |
|-----------|-----------|--------|------|
| ASR | whisper.cpp (tiny.en) | ggml binary | ~75 MB |
| NMT (fast) | Marian OPUS-MT EN→HI | ONNX INT8 | ~534 MB |
| LLM (smart) | Qwen3-0.6B | ExecuTorch .pte INT4 | ~388 MB |
| TTS | Piper VITS Hindi | ONNX | ~17 MB |
| LLM Runtime | ExecuTorch 0.7+ | XNNPACK + KleidiAI | — |
| NMT/TTS Runtime | ONNX Runtime | CPU | — |

---

## Current Status

### ✅ Phase 1: Mac Host Build — COMPLETE
- [x] Fixed macOS compatibility in CMake (5 files: `-march` guard, `.dylib`/`.so`, `nproc`, pthread)
- [x] Initialized whisper.cpp submodule
- [x] Set up conda `deer-arm` environment (Python 3.10, ONNX Runtime, SentencePiece C++)
- [x] Downloaded ASR model (Whisper tiny.en, 74 MB)
- [x] Downloaded/exported MT model (Marian OPUS-MT EN→HI, encoder 194 MB + decoder 340 MB)
- [x] Downloaded TTS model (Piper Hindi VITS, 60 MB)
- [x] Built `translation_host` binary (NMT-only mode)
- [x] Tested: English WAV → Hindi WAV pipeline working end-to-end

### ✅ Phase 2: ExecuTorch + LLM — COMPLETE
- [x] Installed ExecuTorch 1.1.0 (pip) + downloaded C++ headers from GitHub
- [x] Downloaded Qwen3-0.6B from HuggingFace
- [x] Exported to .pte with INT4 quantization (8da4w, group_size 128, XNNPACK backend)
- [x] Verified .pte model via Python runtime — 72-88 tok/s on Apple Silicon M-series
- [x] Tested all three modes: SPEED (NMT ~20ms), BALANCED (speculative ~150ms), QUALITY (LLM ~500ms)
- [x] Model size: 388 MB as .pte (INT4 quantized)

### ✅ Phase 3: Android — COMPLETE
- [x] Installed Android SDK 34, NDK 25.2.9519653, CMake 3.22.1
- [x] Downloaded ONNX Runtime Android AAR v1.22.0, set up arm64-v8a libs
- [x] Built APK (727 MB with all models including NMT)
- [x] ExecuTorch AAR (org.pytorch:executorch-android:0.7.0) integrated
- [x] Tested on ARM64 emulator (Pixel 7, Android 14) — app launches, models extract, shows "Ready"
- [x] All 12 model files extracted successfully (0 failures)
- [x] Native libraries verified: libonnxruntime.so, libnative-lib.so, libexecutorch_jni.so, libwhisper.so

### ✅ Phase 4: Submission — IN PROGRESS
- [x] Project report written
- [ ] Demo video (2 min, show all 3 modes)
- [ ] Package submission per challenge requirements
- [ ] Submit by Feb 20, 2026 11:59 PM IST

---

## Repository Structure

```
arm_s2s-main/
├── asr/                          # whisper.cpp ASR wrapper
│   ├── asr_wrapper.cpp/.h
│   └── whisper_cpp/              # git submodule (needs init)
├── mt/                           # Translation (NMT + LLM)
│   ├── marian_mt_wrapper.cpp/.h  # Marian NMT ONNX backend
│   ├── mt_wrapper.cpp/.h         # Routing: speed/balanced/quality modes
│   ├── llm_translator.cpp/.h     # Speculative decoding + autoregressive LLM
│   └── translation_mode.h        # SPEED/BALANCED/QUALITY enum
├── tts/                          # Piper VITS TTS wrapper
├── pipeline/                     # Threading, queues, orchestration
│   ├── pipeline.cpp/.h           # Main orchestrator
│   ├── lockfree_queue.h          # SPSC queue
│   ├── phrase_detector.cpp/.h    # Detects phrase boundaries
│   └── performance_tracker.cpp/.h
├── host/                         # Desktop CLI app
│   ├── main.cpp                  # Entry point with --llm-model, --translation-mode
│   └── CMakeLists.txt
├── android/                      # Android app
│   ├── app/build.gradle
│   └── native-lib/native-lib.cpp # JNI bridge
├── scripts/
│   ├── build_host.sh
│   ├── download_asr_model.sh
│   ├── download_mt_model.sh
│   ├── download_tts_model.sh
│   ├── download_llm_model.sh     # Qwen3-0.6B from HuggingFace
│   ├── export_llm_model.sh       # Export to .pte
│   └── prepare_models_android.sh
├── tests/
│   └── test_speculative.cpp      # Standalone speculative decoding test
├── models/                       # Downloaded models go here
│   ├── asr/ggml-tiny.en.bin
│   ├── mt/onnx/encoder_model.onnx (+ decoder, tokenizer)
│   ├── tts/hi_IN-rohan-medium.onnx
│   └── llm/qwen3_0.6B.pte
└── docs/ARCHITECTURE.md
```

---

## Key Design Decisions

1. **Pipeline unchanged**: LLM integration is entirely inside `MTWrapper::translate()`. No thread changes, no queue changes. The pipeline just calls `translate()` and gets Hindi text.

2. **Compile flags**: `USE_EXECUTORCH` and `USE_ONNXRUNTIME` are independent `#ifdef` guards. Project works with any combination:
   - Both off → placeholder `[HI: text]`
   - ONNX only → Marian NMT
   - ExecuTorch only → LLM autoregressive
   - Both on → speculative decoding (NMT draft + LLM verify)

3. **Speculative decoding uses low-level Module API**: We call `Module::forward()` directly (not `TextLlmRunner::generate()`) because we need raw logits for token verification. This means we manage KV cache manually.

4. **Three modes are a runtime setting**: Not a compile-time choice. The user/app can switch between Speed/Balanced/Quality at any time via `set_translation_mode()`.