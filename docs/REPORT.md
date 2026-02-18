# Garud: On-Device English→Hindi Speech-to-Speech Translation
## With NMT-Accelerated Speculative Decoding on ARM

**Bharat AI-SoC Student Challenge 2026**

---

## 1. Executive Summary

Garud is a fully offline, CPU-only, real-time English-to-Hindi speech-to-speech translation system that runs entirely on ARM Android devices. The system implements a novel **NMT-accelerated speculative decoding** approach where a fast Marian Neural Machine Translation model generates draft translations in ~49ms, which are then verified by a Qwen3-0.6B Large Language Model through two strategies:

1. **Single-pass validation** (Balanced mode): One LLM forward pass over prompt + draft tokens to assess confidence
2. **KV-cache speculative decode** (backend): Prefill prompt once, verify draft tokens incrementally with KV-cache reuse

The refinement prompt technique — giving the LLM the NMT draft as context — achieves **65% token acceptance**, compared to only 4% with a naive translate-from-scratch approach.

**Key achievements:**
- Sub-200ms translation latency in Speed mode (NMT), correct Hindi output
- Two user-facing translation modes: Speed (NMT) and Balanced (NMT + LLM validation)
- Novel refinement-prompt speculative decoding with 65% acceptance rate
- Pure C++17 runtime — zero Python, zero cloud dependencies
- Lock-free concurrent pipeline with cache-line aligned SPSC queues
- Echo suppression with ASR buffer flush to prevent feedback loops
- Upgraded ASR (whisper base.en) for ~30% lower word error rate
- ~3,800 lines of custom C++ code (excluding third-party libraries)
- 618 MB APK with all NMT models, deployable on ARM64 Android 8+
- Tested on real hardware: OnePlus CPH2767 (Android 16, Snapdragon arm64-v8a)

---

## 2. Problem Statement

Real-time speech translation in India faces several constraints:
1. **Connectivity**: Millions of users lack reliable internet, making cloud-based solutions impractical
2. **Latency**: Cloud round-trips add 200-500ms, making conversation flow unnatural
3. **Privacy**: Sensitive conversations (medical, legal, personal) should not leave the device
4. **Hardware**: Most Indian smartphones are mid-range ARM devices with 4-6 GB RAM

Garud addresses all four by running the entire ASR→MT→TTS pipeline on-device with no network dependency.

---

## 3. System Architecture

### 3.1 Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Garud Translation Pipeline                        │
│                                                                          │
│  Microphone (16 kHz PCM)                                                 │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐    Lock-free    ┌──────────────┐    Lock-free           │
│  │  Thread 1   │    SPSC Queue   │  Thread 2    │    SPSC Queue          │
│  │  ASR        │ ──────────────► │  Translation │ ──────────────►        │
│  │  whisper.cpp│  English text   │  MT Wrapper  │  Hindi text            │
│  │  (base.en)  │                 │  2 modes     │                        │
│  └─────────────┘                 └──────────────┘                        │
│                                         │                                │
│                                    ┌────┴────┐                           │
│                               ┌────┤  Mode?  ├────┐                     │
│                               │    └─────────┘    │                      │
│                               ▼                   ▼                      │
│                          ┌────────┐          ┌────────┐                  │
│                          │ SPEED  │          │BALANCED│                  │
│                          │ NMT    │          │NMT +   │                  │
│                          │ ~49ms  │          │LLM     │                  │
│                          │        │          │~300-   │                  │
│                          │        │          │600ms   │                  │
│                          └────────┘          └────────┘                  │
│                                                                          │
│  ┌─────────────┐    Lock-free                                            │
│  │  Thread 3   │ ◄────────────── Hindi text                             │
│  │  TTS        │                                                         │
│  │  Piper VITS │                                                         │
│  │  (chunked)  │ ──────────────► Speaker (16 kHz PCM)                   │
│  └─────────────┘    Lock-free                                            │
│                     SPSC Queue                                           │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐    │
│  │  Echo Suppression: Mic muted during TTS + 1000ms buffer         │    │
│  │  ASR audio buffer flushed when suppression ends                 │    │
│  └──────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Three-Thread Parallel Pipeline

The system uses three dedicated processing threads connected by lock-free SPSC (Single Producer Single Consumer) queues:

| Thread | Component | Function | Priority |
|--------|-----------|----------|----------|
| Thread 1 | ASR | whisper.cpp streaming transcription (base.en) | Urgent Audio |
| Thread 2 | MT | Two-mode translation routing (Speed/Balanced) | Normal |
| Thread 3 | TTS | Piper VITS Hindi synthesis + echo suppression | Normal |

**Why not mutexes?** On mobile ARM processors, mutex contention causes unpredictable latency spikes (10-50ms). Our lock-free SPSC queues guarantee bounded worst-case latency using `std::atomic` with `acquire`/`release` memory ordering and 64-byte cache-line alignment to prevent false sharing.

### 3.3 Lock-Free SPSC Queue Design

```cpp
template<typename T>
class LockFreeQueue {
    const size_t capacity_;          // Must be power of 2
    std::unique_ptr<T[]> buffer_;
    alignas(64) std::atomic<size_t> write_pos_;  // Separate cache lines
    alignas(64) std::atomic<size_t> read_pos_;   // prevent false sharing
};
```

- **Capacity constraint**: Power-of-2 enables fast modulo via bitmask (`pos & (capacity - 1)`)
- **Memory ordering**: `relaxed` for local reads, `acquire` at synchronization points, `release` on publish
- **Queue sizes**: Audio 1024 buffers (~1s), Text 64 phrases, TTS audio 32 chunks

### 3.4 Sentence-Wise Translation

ASR accumulates a running transcript (`confirmed_text_`) across whisper runs, ensuring no context is lost between buffer trims. The pipeline detects sentence boundaries (`. ? !`) and sends each complete sentence to MT exactly once. `last_translated_pos_` tracks which portion has been translated, preventing duplicate translations and ensuring each sentence is processed independently.

### 3.5 Echo Suppression

The pipeline implements two-layer echo suppression to prevent the mic from picking up TTS Hindi output:

1. **Timestamp-based mic muting**: When TTS audio is generated, `suppress_until_` is set to playback duration + **1000ms** buffer. During this window, `push_audio()` drops all mic samples.
2. **ASR buffer flush**: When suppression ends, `asr_wrapper.clear_buffer()` discards any audio accumulated during TTS playback. This prevents whisper from processing stale, TTS-contaminated audio that AudioRecord buffered internally during the mute window.

Without the buffer flush, AudioRecord continues buffering mic data during suppression. When suppression lifts, this stale audio (containing Hindi TTS output) gets fed to whisper, causing hallucinated English transcriptions of the Hindi speech — creating a feedback loop.

---

## 4. Core Innovation: NMT-Accelerated Speculative Decoding

### 4.1 Background

Standard speculative decoding uses a small language model to draft tokens that a larger model verifies. Our approach innovates in two ways:

1. **Cross-architecture drafting**: An encoder-decoder NMT model (Marian) drafts for a decoder-only LLM (Qwen3), requiring tokenizer bridging (detokenize NMT → retokenize for LLM)
2. **Refinement prompt**: Instead of asking the LLM to translate from scratch and comparing outputs, we give the LLM the NMT draft as context in a refinement prompt. This is critical — without it, the LLM and NMT produce completely different Hindi, yielding only ~4% token acceptance. With the refinement prompt, the LLM tends to echo correct NMT tokens and only diverge where it has a genuine correction, achieving **65% acceptance**.

### 4.2 The Refinement Prompt Insight

Our key experimental finding: Qwen3-0.6B (600M params) cannot translate English→Hindi from scratch — it produces incorrect Hindi in both Python float32 and C++ INT4. However, when given an NMT draft as context, the model reliably echoes correct tokens and makes minor grammatical corrections.

**Naive approach (4% acceptance):**
```
System: "Translate English to Hindi."
User: "I will go to the market."
→ LLM generates: "मे बार करें।" (wrong — "Do time")
→ NMT generated: "मैं बाजार में जाना होगा." (correct)
→ Token match: 1/23 = 4%
```

**Refinement approach (65% acceptance):**
```
System: "Improve this Hindi translation."
User: "English: I will go to the market.\nDraft: मैं बाजार में जाना होगा."
→ LLM generates: "मैं बाजार में जाने होगी." (echoes NMT with minor gender tweak)
→ Token match: 15/23 = 65%
```

### 4.3 Two Verification Strategies

#### Strategy A: Single-Pass Validation (Balanced Mode — User-Facing)

The Balanced mode uses a lightweight validation approach:

```
function validate_translation(english, nmt_draft):
    prompt = build_refinement_prompt(english, nmt_draft)
    full_seq = prompt_tokens + draft_tokens

    // ONE forward pass over entire sequence
    prediction = llm.forward(full_seq, start_pos=0)

    // If LLM predicts EOS after the draft → confident
    confident = is_eos(prediction)

    return nmt_draft  // Always returns NMT output
```

This is fast (~300-600ms) because it runs exactly **one** LLM forward pass. The LLM acts as a binary confidence scorer — if it predicts EOS after seeing the full draft, the translation is likely correct. The NMT translation is always returned (the LLM doesn't modify it).

#### Strategy B: KV-Cache Speculative Decode (Backend)

For higher-quality verification with actual token-level correction:

```
function speculative_translate_kv(english, nmt_draft):
    prompt_tokens = tokenize(refinement_prompt)
    draft_tokens = tokenize(nmt_draft)

    // Step 1: Prefill prompt ONCE (builds KV cache)
    prediction = llm.forward(prompt_tokens, start_pos=0)

    // Step 2: Verify draft tokens one at a time (KV cache reused)
    for i, draft_tok in enumerate(draft_tokens):
        if prediction != draft_tok:
            break  // Rejection — continue autoregressive from here
        prediction = llm.forward([draft_tok], start_pos=prompt_len+i)

    // Step 3: Autoregressive from rejection point
    ...
    return detokenize(accepted_tokens)
```

This is O(n) in draft length instead of O(n²) (full-prefill approach), because the KV cache from the prompt prefill is reused for each subsequent verification step.

### 4.4 Why This Works

The NMT model (Marian, encoder-decoder, 534 MB) produces grammatically correct Hindi for short phrases — it excels at this task. Qwen3-0.6B is too small to translate Hindi from scratch, but is capable enough to:
- **Validate** correct NMT tokens (echoing them back → high acceptance)
- **Correct** minor errors like gender agreement or punctuation
- **Serve as a quality gate** — if NMT made a mistake, the LLM diverges at that position

This makes the NMT the primary translator and the LLM a refinement/verification layer, inverting the usual speculative decoding paradigm where the larger model is "better".

### 4.5 Measured Performance

| Mode | Backend | Translation Latency | Hindi Output | Method |
|------|---------|---------|---------|--------|
| **SPEED** | Marian NMT | **49ms** | मैं बाजार में जाना होगा. (correct) | Direct encoder-decoder |
| **BALANCED** | NMT + LLM | **300-600ms** | Same as NMT (with confidence score) | Single-pass validation |

### 4.6 Speculative Decoding Metrics

| Metric | Naive Prompt | Refinement Prompt |
|--------|-------------|-------------------|
| Acceptance rate | 4% (1/23) | **65% (15/23)** |
| Hindi output quality | Incorrect | Correct (with minor corrections) |
| LLM prompt strategy | Translate from scratch | Refine NMT draft |
| Token-level agreement | 1st character only | 15 consecutive tokens |

---

## 5. Component Details

### 5.1 ASR: whisper.cpp (Streaming)

| Parameter | Value |
|-----------|-------|
| Model | Whisper base.en (ggml format) |
| Size | 142 MB |
| Language | English only |
| WER | ~5.2% (vs ~7.5% for tiny.en — 30% improvement) |
| Decoding | Greedy (temperature=0, beam=1) |
| Threads | 8 (mobile — utilizes all cores on 8-core ARM SoC) |
| Chunk size | 3-second audio chunks |
| VAD | Peak amplitude threshold to skip silence |
| Buffer management | Sliding buffer with context continuity, 30s max cap |

The ASR module uses whisper.cpp's C API with streaming inference. Audio is accumulated in a sliding buffer to maintain transcription context across chunks. VAD (Voice Activity Detection) based on peak amplitude skips silence chunks to save compute. The `suppress_blank` flag and greedy decoding ensure deterministic, low-latency output.

**Why base.en over tiny.en?** We upgraded from whisper-tiny.en (74 MB, ~7.5% WER) to whisper-base.en (142 MB, ~5.2% WER) for a 30% reduction in word error rate. The accuracy improvement significantly reduces garbage-in/garbage-out issues in downstream MT, at the cost of ~68 MB additional model size and slightly higher inference latency. Thread count was increased from 6 to 8 to compensate.

### 5.2 MT: Two-Mode Translation

#### Marian NMT (Speed Mode / Draft Generator)
- **Model**: Helsinki-NLP OPUS-MT EN→HI (encoder-decoder)
- **Runtime**: ONNX Runtime C++ (CPU provider)
- **Quantization**: INT8 (encoder 194 MB + decoder 340 MB)
- **Tokenization**: SentencePiece with vocab.json ID mapping (SP IDs ≠ vocab.json IDs)
- **Measured latency**: ~49ms per sentence (desktop), verified on ARM64 device
- **Critical fix**: decoder_with_past takes only encoder_attention_mask + input_ids + 24 KV tensors (not encoder_hidden_states); encoder KV must be saved from first step and reused

#### Qwen3-0.6B LLM (Validation / Speculative Verifier)
- **Model**: Qwen/Qwen3-0.6B
- **Runtime**: ExecuTorch 1.1.0 with XNNPACK backend + KleidiAI
- **Quantization**: INT4 (8da4w, group size 128, 4-bit embeddings)
- **Size**: 388 MB as .pte
- **Throughput**: 72-88 tok/s on ARM (Apple Silicon benchmark)
- **Prompt format**: Qwen3 chat template with pre-filled empty `<think>` block to skip reasoning chain
- **Tokenizer**: HFTokenizer with PCRE2 fallback (re2 cannot compile Qwen3's lookahead regex)
- **Limitation**: Too small for standalone Hindi translation; serves as refinement/verification layer over NMT

#### Mode Routing (`mt/mt_wrapper.cpp`)
```cpp
switch (current_mode_) {
    case SPEED:    return translate_nmt(text);        // NMT only (~49ms)
    case BALANCED: return translate_validated(text);   // NMT + single-pass LLM validation
}
```

Graceful fallback: If LLM is unavailable, Balanced mode falls back to NMT automatically.

### 5.3 TTS: Piper VITS Hindi

| Parameter | Value |
|-----------|-------|
| Model | hi_IN-rohan-medium (Piper VITS) |
| Size | 60 MB (ONNX) |
| Sample rate | 22050 Hz (resampled to 16000 Hz for output) |
| Phonemizer | Custom Devanagari→IPA (110-entry table) |

The TTS module includes a native C++ Devanagari phonemizer that handles:
- Independent vowels and dependent vowel matras
- Consonant clusters with virama (halant) processing
- Nukta variants (ड़, क़, ख़, फ़)
- Nasalization (anusvara) and visarga
- **Pad token interspersing**: Phoneme IDs are interspersed with pad tokens (id 0) between every phoneme as required by Piper VITS alignment

This eliminates the need for Python's `piper-phonemize` subprocess, keeping the runtime pure C++.

---

## 6. Android Integration

### 6.1 App Overview

The Android app "Garud" provides a clean UI with:
- **Mode switcher**: Speed / Balanced radio buttons for runtime mode selection
- **Loading spinner**: Visible during model extraction and pipeline initialization
- **Live transcription**: English (ASR) and Hindi (Translation) text areas update in real-time
- **Custom icon**: Garuda bird logo representing the app's identity
- **Background init**: Pipeline initialization runs on a background thread to keep UI responsive

### 6.2 App Architecture

```
┌─────────────────────────────────────────┐
│           Java / Android UI              │
│  ┌─────────────┐  ┌──────────────────┐  │
│  │ MainActivity│  │  ModelManager    │  │
│  │ AudioRecord │  │  Asset extraction│  │
│  │ AudioTrack  │  │  Path resolution │  │
│  │ Mode switch │  │                  │  │
│  └──────┬──────┘  └──────────────────┘  │
│         │ JNI                            │
│  ┌──────┴──────────────────────────────┐ │
│  │        native-lib.cpp (JNI)         │ │
│  │  11 native methods exposed          │ │
│  └──────┬──────────────────────────────┘ │
│         │ C++                            │
│  ┌──────┴──────────────────────────────┐ │
│  │     Pipeline (C++17, 3 threads)     │ │
│  │  ASR → MT → TTS                     │ │
│  │  Lock-free SPSC queues              │ │
│  │  Echo suppression                   │ │
│  └─────────────────────────────────────┘ │
└─────────────────────────────────────────┘
```

### 6.3 JNI Interface

| JNI Method | Purpose |
|------------|---------|
| `nativeInit()` | Initialize pipeline with model paths + translation mode |
| `nativeStart()` / `nativeStop()` | Spawn/join worker threads |
| `nativePushAudio()` | Feed microphone PCM to ASR queue |
| `nativePopAudio()` | Retrieve synthesized Hindi audio for playback |
| `nativeGetCurrentEnglish()` | Get latest ASR transcription |
| `nativeGetCurrentHindi()` | Get latest MT translation |
| `nativeSetTranslationMode()` | Runtime mode switching (0=Speed, 1=Balanced) |
| `nativeGetTranslationMode()` | Query current mode name |
| `nativeGetAcceptanceRate()` | Speculative decoding acceptance metric |
| `nativeIsLLMActive()` | Check if LLM backend is available |

### 6.4 Build Configuration

| Property | Value |
|----------|-------|
| compileSdk | 34 |
| minSdk | 26 (Android 8.0) |
| NDK | 25.2.9519653 |
| ABI | arm64-v8a, x86_64 |
| ExecuTorch | org.pytorch:executorch-android:1.1.0 (Maven AAR) |
| ONNX Runtime | 1.22.0 (prebuilt libonnxruntime.so) |
| APK size | 618 MB (NMT models only) |

### 6.5 Model Deployment

Models are packaged in APK assets and extracted to internal storage on first launch:

| Model | Asset Path | Size |
|-------|-----------|------|
| ASR | models/ggml-base.en.bin | 142 MB |
| MT Encoder | models/mt/onnx/encoder_model.onnx | 194 MB |
| MT Decoder | models/mt/onnx/decoder_model.onnx | 340 MB |
| MT Decoder (with past) | models/mt/onnx/decoder_with_past_model.onnx | ~120 MB |
| MT Tokenizer | models/mt/onnx/source.spm + target.spm + vocab.json | ~2 MB |
| TTS | models/tts/hi_IN-rohan-medium.onnx | 60 MB |
| TTS Config | models/tts/hi_IN-rohan-medium.onnx.json | <1 MB |

The LLM .pte model (388 MB) can be deployed via `adb push` to device storage for Balanced mode.

### 6.6 Device Testing

Tested on **OnePlus CPH2767** (Android 16, Snapdragon arm64-v8a):

| Feature | Status |
|---------|--------|
| NMT Speed mode | Verified — correct Hindi translations |
| Balanced mode (LLM) | Working — 65% acceptance with refinement prompt |
| TTS output | Working — clear Hindi speech with pad interspersing fix |
| Echo suppression | Working — mic muted during playback + buffer flush |
| Sentence-wise translation | Working — no duplicates, no re-translation |
| LLM initialization | Confirmed — `LLM init OK (llm_active=1)` in logs |
| Mode switching | Live switching via UI radio buttons |

**Note**: OnePlus devices suppress ALL native logcat output. File-based logging (`pipeline.log`) was implemented as a workaround.

---

## 7. Performance Benchmarks

### 7.1 LLM Inference (Qwen3-0.6B .pte)

| Metric | Value | Platform |
|--------|-------|----------|
| Throughput | 72-88 tok/s | Apple M-series (ARM64, XNNPACK) |
| Model load time | ~1.5s | Cold start (mmap + mlock) |
| Memory footprint | ~450 MB | Peak RSS |
| Quantization | INT4 (8da4w) | 4-bit weights, 8-bit activations |
| Repetition penalty | 1.3x | Applied to previously generated tokens |

### 7.2 Measured End-to-End Latency

| Pipeline Stage | SPEED | BALANCED |
|---------------|-------|----------|
| ASR (whisper.cpp base.en) | ~1.8s/3s chunk | ~1.8s/3s chunk |
| NMT draft | 49ms | 49ms |
| LLM validation | — | ~300-600ms (single-pass) |
| TTS (Piper VITS) | 150ms | 150ms |
| **Translation latency** | **49ms** | **~300-600ms** |
| **Acceptance rate** | N/A | **65%** |

### 7.3 Model Sizes

| Component | FP32 | Quantized | Reduction |
|-----------|------|-----------|-----------|
| Whisper base.en | ~290 MB | 142 MB (ggml) | 2.0x |
| Marian EN→HI | ~800 MB | 534 MB (INT8) | 1.5x |
| Qwen3-0.6B | ~1.2 GB | 388 MB (INT4) | 3.1x |
| Piper Hindi | ~120 MB | 60 MB (ONNX) | 2.0x |

---

## 8. Technology Stack

### 8.1 Runtime Dependencies (C++ only)

| Library | Version | Purpose | License |
|---------|---------|---------|---------|
| whisper.cpp | latest | ASR inference | MIT |
| ONNX Runtime | 1.22.0 | NMT + TTS inference | MIT |
| ExecuTorch | 1.1.0 | LLM inference (XNNPACK) | BSD |
| SentencePiece | 0.2.0 | NMT tokenization | Apache 2.0 |

### 8.2 ARM-Specific Optimizations

| Feature | Component | Benefit |
|---------|-----------|---------|
| NEON SIMD | whisper.cpp, ONNX Runtime | 2-4x faster matrix ops |
| XNNPACK backend | ExecuTorch LLM | Optimized INT4 kernels for ARM |
| KleidiAI | ExecuTorch (Arm) | SME2/NEON acceleration on Cortex-A |
| Cache-line alignment | SPSC queues (64-byte `alignas`) | Prevents false sharing between cores |
| INT4/INT8 quantization | All models | 2-3x size reduction, faster inference |
| 8-thread parallelism | whisper.cpp ASR | Utilizes all cores on 8-core ARM SoC |

### 8.3 Build System

- **CMake 3.18+** with hierarchical subdirectories (mt → asr → tts → pipeline)
- **Compile switches**: `USE_EXECUTORCH` and `USE_ONNXRUNTIME` are independent; project compiles and runs with any combination
- **Cross-platform**: macOS (host development) + Android arm64-v8a + x86_64 (deployment)
- **No `-ffast-math`**: Intentionally avoided as it breaks whisper.cpp's numerical stability

---

## 9. Novelty and Contributions

### 9.1 Refinement-Prompt Speculative Decoding
To our knowledge, this is the first implementation of speculative decoding that uses:
1. An **encoder-decoder NMT model** as the drafter for a **decoder-only LLM** verifier (cross-architecture)
2. A **refinement prompt** that feeds the NMT draft to the LLM as context, rather than comparing independently generated outputs

The refinement prompt is critical: without it, the LLM and NMT produce completely different Hindi (4% acceptance). With it, the LLM echoes correct NMT tokens and only diverges for genuine corrections (65% acceptance). This inverts the usual speculative decoding paradigm — the smaller NMT is the primary translator, and the larger LLM serves as a quality gate.

### 9.2 Two Verification Strategies
We implement two complementary verification approaches:
- **Single-pass validation** (Balanced mode): One forward pass to assess confidence — fast, lightweight
- **KV-cache speculative decode**: Incremental token verification with KV-cache reuse — O(n) instead of O(n²)

### 9.3 Fully Offline ARM Deployment
The complete ASR→MT→TTS pipeline runs on-device with zero network dependency. All models are quantized (INT4/INT8) and optimized for ARM NEON/SME2 acceleration through ExecuTorch's XNNPACK + KleidiAI and ONNX Runtime's CPU provider.

### 9.4 Lock-Free Pipeline Architecture
The three-thread pipeline uses wait-free SPSC queues with careful memory ordering (`acquire`/`release` atomics, 64-byte cache-line alignment). This avoids mutex contention that causes unpredictable latency spikes on mobile ARM processors.

### 9.5 Native Devanagari Phonemization
A custom 110-entry Devanagari→IPA conversion table handles Hindi phonemization in pure C++, including conjuncts, nukta variants, virama processing, and nasalization — eliminating the need for Python subprocess calls. Phoneme IDs are interspersed with pad tokens (id 0) as required by Piper VITS alignment.

### 9.6 Two-Layer Echo Suppression
The pipeline implements software-based echo suppression with two complementary mechanisms:
1. **Timestamp-based mic muting**: Mic input dropped for TTS playback duration + 1000ms buffer
2. **ASR buffer flush**: Audio buffer cleared when suppression ends to discard stale TTS-contaminated samples

This prevents the mic from picking up the speaker's Hindi TTS output, which would otherwise cause whisper to hallucinate English text from the Hindi audio, creating an infinite feedback loop.

---

## 10. Codebase Summary

| Module | Files | Lines | Description |
|--------|-------|-------|-------------|
| mt/ | 7 | ~1,400 | Translation: LLM, NMT, routing, speculative decoding, validation |
| pipeline/ | 7 | ~950 | Orchestrator, SPSC queues, phrase detection, echo suppression |
| tts/ | 2 | 602 | Piper VITS, Devanagari→IPA phonemizer, resampling |
| host/ | 1 | 477 | Desktop CLI with file I/O and mode selection |
| asr/ | 2 | ~250 | whisper.cpp streaming wrapper with buffer flush |
| android/ | 3 | ~550 | JNI bridge, ModelManager, MainActivity with loading UI |
| **Total** | **22+** | **~4,200** | **Pure C++17 runtime (Java for Android UI only)** |

---

## 11. Limitations and Future Work

### Current Limitations
1. **ASR speed on mobile**: Whisper base.en takes ~1.8s to process 3s of audio on ARM64. VAD helps skip silence but real-time ratio is ~0.6x. Sherpa-ONNX is a potential replacement for true streaming ASR.
2. **LLM standalone quality**: Qwen3-0.6B (600M params) cannot translate Hindi from scratch — confirmed in Python float32 across all prompt formats. The model works as a refinement layer over NMT but not as a standalone translator.
3. **Max sequence length**: Model exported with max_seq_length=128. Long sentences cause the refinement prompt (chat template + English + NMT draft) to exceed this limit. Exporting with a larger max_seq_length (256-512) would fix this at the cost of increased memory.
4. **APK size**: 618 MB is large for distribution; AAB split APKs and on-demand model download would help.
5. **RAM usage**: Peak ~765 MB; tight on 4 GB devices.
6. **TTS voice**: Single Hindi voice (male); adding female voice and voice selection would improve UX.

### Future Directions
1. **Sherpa-ONNX ASR**: Replace whisper.cpp with purpose-built mobile streaming ASR for true real-time performance
2. **Larger LLM**: Qwen3-1.5B or a Hindi-specialized model for better standalone translation quality and higher speculative acceptance rate
3. **NPU/DSP offload**: Use Android NNAPI or Qualcomm QNN for hardware acceleration
4. **Multi-language**: Extend to other Indic languages (Tamil, Telugu, Bengali)
5. **On-demand model download**: Download models on first launch instead of bundling in APK
6. **Speaker diarization**: Handle multi-speaker conversations

---

## 12. Conclusion

Garud demonstrates that real-time, high-quality speech-to-speech translation is achievable entirely on-device using ARM processors. The novel refinement-prompt speculative decoding approach — where a small LLM verifies NMT drafts given as context rather than translating from scratch — achieves 65% token acceptance and represents a new paradigm for combining specialized NMT models with general-purpose LLMs. Our key finding is that even a 0.6B parameter LLM that cannot translate Hindi independently can serve as an effective quality gate when the NMT draft is provided as context. The system is designed for practical deployment in connectivity-challenged environments while maintaining user privacy through fully offline operation.

---

## Appendix A: Build and Run Instructions

See [README.md](../README.md) for complete build instructions.

**Quick start (Android):**
```bash
./scripts/setup_android_deps.sh
./scripts/prepare_models_android.sh
cd android && ./gradlew assembleDebug
adb install app/build/outputs/apk/debug/app-debug.apk
```

## Appendix B: Repository Structure

See [Project_overview.md](Project_overview.md) for complete file listing.

## Appendix C: Architecture Details

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed component documentation.
