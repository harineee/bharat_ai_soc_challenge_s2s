# TODO — ARMM Optimization & Remaining Work

## Priority 1: ASR Speed Optimization (Critical)

Current state: Whisper-tiny.en on ARM64, 4 threads, 2-second chunks. ~4 seconds to process 2 seconds of audio (0.5x real-time).

### Quick Wins (config/model swaps, minimal code changes)

- [ ] **Quantized whisper model** — Use `ggml-tiny.en-q5_1.bin` instead of fp16.
  Smaller model = faster matrix math. Whisper.cpp supports this natively, just swap the file.
  Expected: 1.5-2x speedup.

- [ ] **`speed_up = true` in whisper params** — Downsamples audio 16kHz → 8kHz internally,
  cutting compute ~2x. Some quality loss on unclear speech but fine for clear input.

- [ ] **Voice Activity Detection (VAD)** — Skip whisper on silence.
  Currently we process every 2-second chunk regardless. Simple amplitude threshold before
  calling whisper avoids wasting compute on silence. Whisper.cpp also has built-in energy VAD.

- [ ] **Increase thread count** — OnePlus has 8 cores (big.LITTLE). Currently using 4 threads.
  Try 6-8. Diminishing returns past the number of big cores, but worth benchmarking.

- [ ] **Larger audio chunks (3-4 seconds)** — Whisper has fixed per-run overhead. Bigger chunks
  = better throughput (fewer runs). Trade-off is higher per-sentence latency, but acceptable
  since we do sentence-wise translation.

### Bigger Changes (more effort, bigger payoff)

- [ ] **Sherpa-ONNX** — Purpose-built for real-time streaming ASR on mobile. Uses ONNX Runtime
  (already available). Supports true streaming mode (incremental processing, no re-processing).
  Would require replacing the ASR module but is the "proper" mobile ASR solution.

- [ ] **Vosk (Kaldi-based)** — Lightweight offline mobile ASR. True streaming (frame-by-frame).
  Older technology, lower accuracy than whisper, but very fast on ARM.

- [ ] **Pin whisper threads to performance cores** — ARM big.LITTLE thread affinity.
  Ensure whisper runs on the big (performance) cores, not efficiency cores.

## Priority 2: Qwen LLM Integration (Demo-Ready)

Current state: Qwen3-0.6B exported as .pte (388MB, INT4). Works in Python (72-88 tok/s on M-series).
NOT active in the Android app — running Speed mode (NMT only). Full-prefill takes 6.5s per sentence.

### Enable on Device (Option B — toggleable feature)

- [ ] **Push .pte model to device** — `adb push models/llm/qwen3_0.6B.pte /data/local/tmp/`
  and pass path to `nativeInit`. ModelManager currently doesn't handle LLM models.

- [ ] **Wire up mode toggle in Android UI** — Add Speed/Balanced/Quality selector.
  JNI `nativeSetTranslationMode` already exists, just need UI buttons/dropdown.

- [ ] **Verify ExecuTorch builds on Android** — Confirm `USE_EXECUTORCH=ON` in the Android
  CMakeLists, and that `org.pytorch:executorch-android:0.7.0` AAR is linked.

- [ ] **Test Balanced mode on device** — NMT draft + LLM verify. Will be ~6.5s per sentence
  (full-prefill). Confirm it produces correct Hindi with the refinement prompt.

### KV-Cache Optimization (stretch goal — high risk)

- [ ] **Fix INT4 KV-cache decode** — Incremental decoding was unreliable with INT4 quantization.
  If fixed, latency drops from 6.5s to ~300-500ms (O(n) instead of O(n²)).
  Requires debugging ExecuTorch's quantized attention with cached KV states.

- [ ] **Re-export .pte with larger max_seq_length** — Currently 128. Refinement prompt
  (chat template + English + NMT draft) can exceed this on long sentences, causing
  ExecuTorch error 16. Re-export with 256-512 at the cost of more memory.

### Bigger Model (future — not for hackathon)

- [ ] **Qwen3-1.5B or Hindi-specialized model** — 0.6B is too small for standalone Hindi
  translation. A larger model would improve speculative acceptance rate beyond 65%
  and potentially enable standalone Quality mode.

## Priority 3: TTS Quality Verification

- [ ] **Verify Piper VITS output quality** — Pad token interspersing fix applied, need to
  confirm audio quality on diverse Hindi sentences.
- [ ] **Test with longer Hindi text** — Ensure phoneme conversion handles complex conjuncts
  and rare characters.

## Priority 4: General Polish

- [ ] **Remove whisper hallucinations** — Filter out common hallucinations like "[MUSIC PLAYING]",
  "[BLANK_AUDIO]", "(speaking in foreign language)" before sending to MT.
- [ ] **Improve ASR name recognition** — whisper-tiny struggles with proper nouns.
  Consider a post-processing dictionary for common Indian names.
- [ ] **UI improvements** — Show sentence-by-sentence translation progress in the Android UI.
  Display both English and Hindi transcripts growing in real-time.

## Priority 5: Next Up — Echo Fix + ASR Upgrade

### Step 1: Fix Echo / Feedback Loop (ASR picking up TTS)

Current echo suppression uses timestamp-based mic muting (`suppress_until_`),
but ASR still picks up TTS audio. Evidence from device logs:
`">> [FOREIGN] It's got her namisari."` — ASR transcribing Hindi TTS as garbled English.

**Root cause:** AudioRecord keeps buffering mic audio internally even while we drop
samples in `push_audio()`. When suppression ends, stale TTS-contaminated audio is
already in the buffer and gets fed to whisper.

**Plan (do both together):**

- [x] **Fix A — Flush ASR audio buffer after TTS playback.** Added `clear_buffer()`
  to `asr/asr_wrapper.h/.cpp`. Called in `pipeline/pipeline.cpp` when suppression ends.

- [x] **Fix B — Increase suppression buffer from +500ms to +1000ms.** Changed in
  `pipeline/pipeline.cpp`. Catches AudioTrack drain delay + room reverb.

### Step 2: Upgrade ASR to whisper-base.en

Current whisper-tiny.en has noticeable quality issues: hallucinations on background
noise, missed quiet words, poor accent handling.

| | tiny.en (now) | base.en |
|---|---|---|
| Model size | 74MB | 142MB |
| WER (word error rate) | ~7.5% | ~5.2% |
| Est. device latency | ~900ms/3s | ~1.8s/3s |
| Real-time ratio | 0.3x | 0.6x |
| APK size impact | — | +68MB |

**Plan:**

- [x] **Download ggml-base.en.bin** (141MB) from HuggingFace whisper.cpp repo
- [x] **Update `scripts/download_asr_model.sh`** to fetch base.en instead of tiny.en
- [x] **Update Android assets** — swapped model, updated ModelManager.java,
  MainActivity.java, NativePipeline.java, prepare_models scripts, build_host.sh
- [x] **Tune params for base.en** — bumped threads 6→8, kept audio_ctx=512, 3s chunks
- [ ] **Test on OnePlus** — verify stays under 2s for 3-4s chunks (real-time capable)
- [ ] **Benchmark** — compare tiny vs base on same test sentences, log timings

## Completed

- [x] Sentence-wise translation (ASR accumulates, translates per sentence, no re-translation)
- [x] Echo suppression (mic muted during TTS playback + 1000ms buffer + ASR buffer flush)
- [x] ASR context accumulation (confirmed_text_ grows across whisper runs)
- [x] Piper VITS pad token interspersing (required for correct alignment)
- [x] Pipeline thread safety (mutex on shared strings)
- [x] Phrase detector duplicate prevention
- [x] INT8 quantized MT models for Android (862MB → 217MB)
- [x] decoder_with_past_model.onnx added to Android assets
- [x] File-based logging for OnePlus devices (logcat suppressed)
- [x] Pipeline stop flush (translate remaining ASR text before shutdown)
- [x] ExecuTorch v1.1.0 compiled into Android APK (USE_EXECUTORCH=ON)
- [x] LLM model pushed to device and initializes successfully (llm_active=1)
- [x] Mode switcher UI (Speed/Balanced/Quality radio buttons)
- [x] KV-cache speculative decode implemented (Quality mode)
- [x] Single-pass LLM validation implemented (Balanced mode)
- [x] Legacy full-prefill speculative kept as fallback
- [x] ASR upgraded from whisper-tiny.en (74MB) to whisper-base.en (141MB, ~30% lower WER)
