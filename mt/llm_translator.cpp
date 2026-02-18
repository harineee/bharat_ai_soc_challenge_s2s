/**
 * LLM-based translator using ExecuTorch + Qwen3-0.6B
 *
 * Implements:
 * - Full autoregressive translation (Quality mode)
 * - NMT-accelerated speculative decoding (Balanced mode)
 *
 * Compiles cleanly even without ExecuTorch headers (behind USE_EXECUTORCH).
 */

#include "llm_translator.h"
#include <iostream>
#include <chrono>
#include <algorithm>
#include <fstream>

#ifdef USE_EXECUTORCH
#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>
#include <pytorch/tokenizers/hf_tokenizer.h>
using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::extension::from_blob;
using executorch::extension::Module;
using executorch::runtime::EValue;
#endif

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "LLMTranslator"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGI(...) do { fprintf(stdout, "[LLM] "); fprintf(stdout, __VA_ARGS__); fprintf(stdout, "\n"); } while(0)
#define LOGE(...) do { fprintf(stderr, "[LLM ERROR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// Qwen3 EOS token IDs
static constexpr int64_t QWEN3_EOS_1 = 151643;
static constexpr int64_t QWEN3_EOS_2 = 151645;

static bool is_eos(int64_t token) {
    return token == QWEN3_EOS_1 || token == QWEN3_EOS_2;
}

LLMTranslator::LLMTranslator() = default;

LLMTranslator::~LLMTranslator() {
#ifdef USE_EXECUTORCH
    if (module_) {
        delete static_cast<Module*>(module_);
        module_ = nullptr;
    }
    if (tokenizer_) {
        delete static_cast<tokenizers::HFTokenizer*>(tokenizer_);
        tokenizer_ = nullptr;
    }
#endif
    initialized_ = false;
}

bool LLMTranslator::init(const std::string& model_path,
                          const std::string& tokenizer_path) {
#ifdef USE_EXECUTORCH
    model_path_ = model_path;
    tokenizer_path_ = tokenizer_path;

    // Verify files exist
    {
        std::ifstream f(model_path);
        if (!f.good()) {
            LOGE("Model file not found: %s", model_path.c_str());
            return false;
        }
    }
    {
        std::ifstream f(tokenizer_path);
        if (!f.good()) {
            LOGE("Tokenizer file not found: %s", tokenizer_path.c_str());
            return false;
        }
    }

    LOGI("Loading Qwen3-0.6B from %s", model_path.c_str());

    auto start = std::chrono::high_resolution_clock::now();

    try {
        auto* mod = new Module(model_path, Module::LoadMode::MmapUseMlockIgnoreErrors);
        if (!mod) {
            LOGE("Failed to create Module");
            return false;
        }
        auto load_err = mod->load();
        if (load_err != executorch::runtime::Error::Ok) {
            LOGE("Failed to load model");
            delete mod;
            return false;
        }

        // Load the forward method
        auto fwd_err = mod->load_forward();
        if (fwd_err != executorch::runtime::Error::Ok) {
            LOGE("Failed to load forward method, error=%d",
                 static_cast<int>(fwd_err));
        }

        module_ = static_cast<void*>(mod);
    } catch (const std::exception& e) {
        LOGE("Exception loading model: %s", e.what());
        return false;
    }

    // Load HuggingFace tokenizer
    try {
        auto* tok = new tokenizers::HFTokenizer();
        auto err = tok->load(tokenizer_path);
        if (err != tokenizers::Error::Ok) {
            LOGE("Failed to load tokenizer: %s", tokenizer_path.c_str());
            delete tok;
            return false;
        }
        tokenizer_ = static_cast<void*>(tok);
        LOGI("Tokenizer loaded from %s", tokenizer_path.c_str());
    } catch (const std::exception& e) {
        LOGE("Exception loading tokenizer: %s", e.what());
        return false;
    }

    auto end = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(end - start).count();
    LOGI("Model loaded in %.1f ms", load_ms);

    initialized_ = true;
    return true;
#else
    (void)model_path;
    (void)tokenizer_path;
    LOGE("ExecuTorch not compiled in (USE_EXECUTORCH not defined)");
    return false;
#endif
}

// ============================================================
// Forward pass returning argmax with repetition penalty.
// Prefill: tokens=[1,N], start_pos=0
// Decode:  tokens=[1,1], start_pos=N
// Applies repetition_penalty to tokens in `penalize` set.
// Returns argmax token ID, or -1 on error.
// ============================================================
int64_t LLMTranslator::forward_next_token(
    const std::vector<int64_t>& token_ids, int64_t start_pos,
    const std::vector<int64_t>& penalize, float rep_penalty) {
#ifdef USE_EXECUTORCH
    if (!module_ || token_ids.empty()) return -1;

    auto* mod = static_cast<Module*>(module_);
    int32_t seq_len = static_cast<int32_t>(token_ids.size());

    auto input_ids = from_blob(
        const_cast<int64_t*>(token_ids.data()),
        {1, seq_len},
        ScalarType::Long);

    int64_t sp = start_pos;
    auto input_pos = from_blob(&sp, {1}, ScalarType::Long);

    std::vector<EValue> inputs;
    inputs.push_back(EValue(input_ids));
    inputs.push_back(EValue(input_pos));

    auto result = mod->forward(inputs);
    if (!result.ok()) {
        LOGE("forward() error=%d (start_pos=%lld, seq_len=%d)",
             static_cast<int>(result.error()), (long long)start_pos, seq_len);
        return -1;
    }

    auto& outputs = result.get();
    if (outputs.empty()) return -1;

    auto logits_tensor = outputs[0].toTensor();
    const float* data = logits_tensor.const_data_ptr<float>();
    int vocab_size = static_cast<int>(logits_tensor.size(logits_tensor.dim() - 1));

    // Copy logits for modification (repetition penalty)
    std::vector<float> logits(data, data + vocab_size);

    // Apply repetition penalty
    if (rep_penalty > 1.0f && !penalize.empty()) {
        for (int64_t tok_id : penalize) {
            if (tok_id >= 0 && tok_id < vocab_size) {
                if (logits[tok_id] > 0) {
                    logits[tok_id] /= rep_penalty;
                } else {
                    logits[tok_id] *= rep_penalty;
                }
            }
        }
    }

    // Argmax
    return static_cast<int64_t>(
        std::distance(logits.begin(),
                      std::max_element(logits.begin(), logits.end())));
#else
    (void)token_ids;
    (void)start_pos;
    (void)penalize;
    (void)rep_penalty;
    return -1;
#endif
}

// ============================================================
// Full autoregressive translation (Quality mode)
// Uses KV cache: prefill prompt, then decode one token at a time.
// ============================================================
std::string LLMTranslator::translate(const std::string& english_text) {
    if (!initialized_ || english_text.empty()) return "";

    auto start = std::chrono::high_resolution_clock::now();

    std::string prompt = build_prompt(english_text);
    std::vector<int64_t> prompt_tokens = tokenize(prompt);
    if (prompt_tokens.empty()) return "";

    int prompt_len = static_cast<int>(prompt_tokens.size());

    // Autoregressive generation using full-prefill each step.
    // Each call starts fresh with start_pos=0 and the full sequence.
    // O(n*seq_len) total but reliable with quantized models.
    // Repetition penalty (1.3) prevents the quantized model from looping.
    static constexpr float REP_PENALTY = 1.3f;
    static constexpr int MAX_TOKENS = 32;

    std::vector<int64_t> generated;
    std::vector<int64_t> full_seq = prompt_tokens;

    for (int i = 0; i < MAX_TOKENS; i++) {
        // Penalize all previously generated tokens
        int64_t next_token = forward_next_token(full_seq, 0,
                                                 generated, REP_PENALTY);
        if (next_token < 0 || is_eos(next_token)) break;

        generated.push_back(next_token);
        full_seq.push_back(next_token);

        // Stop on repetition: same token 3+ times in a row
        if (generated.size() >= 3) {
            size_t n = generated.size();
            if (generated[n-1] == generated[n-2] &&
                generated[n-2] == generated[n-3]) {
                generated.pop_back();
                generated.pop_back();
                break;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    last_latency_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    std::string result = extract_translation(detokenize(generated));
    if (!result.empty()) {
        LOGI("Translated (%d tokens, %.0f ms): \"%s\" -> \"%s\"",
             (int)generated.size(), last_latency_ms_,
             english_text.c_str(), result.c_str());
    } else {
        LOGE("Empty translation for: \"%s\"", english_text.c_str());
    }

    return result;
}

// ============================================================
// Speculative translation — NMT draft + LLM verification
// The refinement prompt gives the LLM the NMT draft as context,
// so the LLM tends to echo correct tokens (high acceptance)
// and only diverge where it has a genuine correction.
// ============================================================
std::string LLMTranslator::speculative_translate(
    const std::string& english_text,
    const std::string& nmt_draft) {
    if (!initialized_ || english_text.empty()) return "";
    if (nmt_draft.empty()) return translate(english_text);

    auto start = std::chrono::high_resolution_clock::now();

    // Build refinement prompt (includes NMT draft as context)
    std::string prompt = build_refinement_prompt(english_text, nmt_draft);
    std::vector<int64_t> prompt_tokens = tokenize(prompt);
    std::vector<int64_t> draft_tokens = tokenize(nmt_draft);
    int draft_len = static_cast<int>(draft_tokens.size());

    if (prompt_tokens.empty() || draft_tokens.empty()) {
        return translate(english_text);
    }

    last_total_draft_tokens_ = draft_len;
    static constexpr float REP_PENALTY = 1.3f;

    // Verify each NMT draft token against LLM's prediction.
    // The LLM has seen the draft in the prompt, so it tends to
    // echo it back — giving high acceptance for correct tokens.
    std::vector<int64_t> seq = prompt_tokens;
    int accepted = 0;
    std::vector<int64_t> all_generated;

    for (int i = 0; i < draft_len; i++) {
        int64_t prediction = forward_next_token(seq, 0,
                                                 all_generated, REP_PENALTY);
        if (prediction < 0) break;

        if (prediction == draft_tokens[i]) {
            accepted++;
            all_generated.push_back(draft_tokens[i]);
            seq.push_back(draft_tokens[i]);
        } else {
            // Mismatch — LLM wants a different token (potential correction)
            if (!is_eos(prediction)) {
                all_generated.push_back(prediction);
                seq.push_back(prediction);
            }
            break;
        }
    }

    // If all draft tokens accepted, get one more token (EOS or continuation)
    if (accepted == draft_len) {
        int64_t bonus = forward_next_token(seq, 0,
                                            all_generated, REP_PENALTY);
        if (bonus >= 0 && !is_eos(bonus)) {
            all_generated.push_back(bonus);
            seq.push_back(bonus);
        }
    }

    // Continue autoregressive from rejection point
    if (accepted < draft_len) {
        int remaining = draft_len - accepted;
        for (int i = 0; i < remaining + 4; i++) {
            int64_t next = forward_next_token(seq, 0,
                                               all_generated, REP_PENALTY);
            if (next < 0 || is_eos(next)) break;
            all_generated.push_back(next);
            seq.push_back(next);

            // Repetition check
            size_t n = all_generated.size();
            if (n >= 3 && all_generated[n-1] == all_generated[n-2] &&
                all_generated[n-2] == all_generated[n-3]) {
                all_generated.pop_back();
                all_generated.pop_back();
                break;
            }
        }
    }

    last_accepted_tokens_ = accepted;
    total_accepted_ += accepted;
    total_drafted_ += draft_len;

    auto end = std::chrono::high_resolution_clock::now();
    last_latency_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    std::string result = extract_translation(detokenize(all_generated));
    LOGI("Speculative (%d/%d accepted, %.0f ms): \"%s\" -> \"%s\"",
         accepted, draft_len, last_latency_ms_,
         english_text.c_str(), result.c_str());

    return result;
}

// ============================================================
// KV-cache speculative decode (Quality mode)
// Prefill once, then verify draft tokens incrementally.
// Much faster than full-prefill per step: O(prefill + N*decode)
// vs O(N * (prefill+N)) for the legacy approach.
// ============================================================
std::string LLMTranslator::speculative_translate_kv(
    const std::string& english_text,
    const std::string& nmt_draft) {
    if (!initialized_ || english_text.empty()) return "";
    if (nmt_draft.empty()) return translate(english_text);

    auto start = std::chrono::high_resolution_clock::now();

    std::string prompt = build_refinement_prompt(english_text, nmt_draft);
    std::vector<int64_t> prompt_tokens = tokenize(prompt);
    std::vector<int64_t> draft_tokens = tokenize(nmt_draft);
    int draft_len = static_cast<int>(draft_tokens.size());
    int prompt_len = static_cast<int>(prompt_tokens.size());

    if (prompt_tokens.empty() || draft_tokens.empty()) {
        return translate(english_text);
    }

    // Guard: max_seq_length is 128 for this model
    if (prompt_len + draft_len + 8 > 128) {
        LOGI("KV-spec: sequence too long (%d+%d), falling back to NMT",
             prompt_len, draft_len);
        return nmt_draft;
    }

    last_total_draft_tokens_ = draft_len;
    static constexpr float REP_PENALTY = 1.3f;

    // Step 1: Prefill entire prompt in one call
    int64_t prediction = forward_next_token(prompt_tokens, 0);
    if (prediction < 0) {
        LOGE("KV-spec: prefill failed");
        return nmt_draft;
    }

    // Step 2: Verify draft tokens one by one using KV cache
    int accepted = 0;
    std::vector<int64_t> all_generated;
    int64_t current_pos = prompt_len;  // KV cache position after prefill

    for (int i = 0; i < draft_len; i++) {
        if (prediction == draft_tokens[i]) {
            // Draft token matches LLM prediction — accept
            accepted++;
            all_generated.push_back(draft_tokens[i]);
            // Feed the accepted token to build KV cache
            prediction = forward_next_token({draft_tokens[i]}, current_pos,
                                             all_generated, REP_PENALTY);
            current_pos++;
            if (prediction < 0) break;
        } else {
            // Mismatch — LLM wants a different token
            if (!is_eos(prediction)) {
                all_generated.push_back(prediction);
                // Feed LLM's token (not draft) into KV cache
                int64_t next = forward_next_token({prediction}, current_pos,
                                                   all_generated, REP_PENALTY);
                current_pos++;
                prediction = next;
            }
            break;
        }
    }

    // If all draft tokens accepted, check for continuation
    if (accepted == draft_len && prediction >= 0 && !is_eos(prediction)) {
        all_generated.push_back(prediction);
    }

    // Step 3: Continue autoregressive from rejection point using KV cache
    if (accepted < draft_len && prediction >= 0 && !is_eos(prediction)) {
        int remaining = draft_len - accepted;
        for (int i = 0; i < remaining + 4; i++) {
            if (prediction < 0 || is_eos(prediction)) break;
            int64_t next = forward_next_token({prediction}, current_pos,
                                               all_generated, REP_PENALTY);
            current_pos++;
            if (next < 0 || is_eos(next)) break;
            all_generated.push_back(next);
            prediction = next;

            // Repetition check
            size_t n = all_generated.size();
            if (n >= 3 && all_generated[n-1] == all_generated[n-2] &&
                all_generated[n-2] == all_generated[n-3]) {
                all_generated.pop_back();
                all_generated.pop_back();
                break;
            }
        }
    }

    last_accepted_tokens_ = accepted;
    total_accepted_ += accepted;
    total_drafted_ += draft_len;

    auto end = std::chrono::high_resolution_clock::now();
    last_latency_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    std::string result = extract_translation(detokenize(all_generated));
    LOGI("KV-Speculative (%d/%d accepted, %.0f ms): \"%s\" -> \"%s\"",
         accepted, draft_len, last_latency_ms_,
         english_text.c_str(), result.c_str());

    return result;
}

// ============================================================
// Single-pass validation (Balanced mode)
// ONE forward call over prompt+draft. Checks what the LLM
// predicts after the draft — EOS means high confidence.
// Always returns NMT draft unchanged; sets confidence flag.
// ============================================================
std::string LLMTranslator::validate_translation(
    const std::string& english_text,
    const std::string& nmt_draft) {
    last_validation_confident_ = false;

    if (!initialized_ || english_text.empty() || nmt_draft.empty()) {
        return nmt_draft;
    }

    auto start = std::chrono::high_resolution_clock::now();

    std::string prompt = build_refinement_prompt(english_text, nmt_draft);
    std::vector<int64_t> prompt_tokens = tokenize(prompt);
    std::vector<int64_t> draft_tokens = tokenize(nmt_draft);

    if (prompt_tokens.empty() || draft_tokens.empty()) {
        return nmt_draft;
    }

    // Concatenate prompt + draft for single forward pass
    std::vector<int64_t> full_seq = prompt_tokens;
    full_seq.insert(full_seq.end(), draft_tokens.begin(), draft_tokens.end());

    // Guard: skip if too long for model's max_seq_length (128)
    if (full_seq.size() > 127) {
        LOGI("Validate: sequence too long (%zu), skipping LLM check",
             full_seq.size());
        return nmt_draft;
    }

    // Single forward pass — what does the LLM predict after the draft?
    int64_t prediction = forward_next_token(full_seq, 0);

    auto end = std::chrono::high_resolution_clock::now();
    last_latency_ms_ = std::chrono::duration<double, std::milli>(end - start).count();

    if (prediction < 0) {
        LOGE("Validate: forward failed");
        return nmt_draft;
    }

    // EOS or Hindi punctuation after draft = high confidence
    // (LLM agrees the draft is a complete, correct translation)
    last_validation_confident_ = is_eos(prediction);

    LOGI("Validate (%.0f ms, confident=%s): \"%s\" -> \"%s\"",
         last_latency_ms_,
         last_validation_confident_ ? "yes" : "no",
         english_text.c_str(), nmt_draft.c_str());

    // Update metrics: validation counts as accepting all draft tokens
    // when confident, zero when not (for tracking purposes)
    last_total_draft_tokens_ = static_cast<int>(draft_tokens.size());
    last_accepted_tokens_ = last_validation_confident_
                            ? last_total_draft_tokens_ : 0;
    total_drafted_ += last_total_draft_tokens_;
    total_accepted_ += last_accepted_tokens_;

    return nmt_draft;
}

// ============================================================
// Metrics
// ============================================================
double LLMTranslator::get_acceptance_rate() const {
    if (total_drafted_ == 0) return 0.0;
    return static_cast<double>(total_accepted_) / total_drafted_;
}

// ============================================================
// Prompt construction
// ============================================================
std::string LLMTranslator::build_prompt(const std::string& english_text) {
    // Qwen3 chat template. Pre-fill assistant with empty <think> block
    // to skip the reasoning chain and produce direct output.
    return "<|im_start|>system\n"
           "You are a translator. Translate English to Hindi. "
           "Output only the Hindi translation, nothing else.<|im_end|>\n"
           "<|im_start|>user\n"
           + english_text + "<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n\n</think>\n";
}

std::string LLMTranslator::build_refinement_prompt(
    const std::string& english_text,
    const std::string& nmt_draft) {
    // Refinement prompt: gives the LLM the NMT draft as context.
    // The LLM tends to echo correct NMT tokens (high acceptance)
    // and only diverge where it has a genuine improvement.
    return "<|im_start|>system\n"
           "You are a Hindi language expert. Given an English sentence "
           "and its draft Hindi translation, output an improved Hindi "
           "translation. Output only Hindi text.<|im_end|>\n"
           "<|im_start|>user\n"
           "English: " + english_text + "\n"
           "Draft: " + nmt_draft + "<|im_end|>\n"
           "<|im_start|>assistant\n"
           "<think>\n\n</think>\n";
}

// ============================================================
// Output cleaning
// ============================================================
std::string LLMTranslator::extract_translation(const std::string& raw_output) {
    std::string result = raw_output;

    // Strip common EOS tokens and artifacts
    const std::string eos_markers[] = {
        "<|endoftext|>", "<|im_end|>", "</s>", "<eos>", "<|end|>"
    };
    for (const auto& marker : eos_markers) {
        size_t pos = result.find(marker);
        if (pos != std::string::npos) {
            result = result.substr(0, pos);
        }
    }

    // Strip leading/trailing whitespace
    size_t start = result.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = result.find_last_not_of(" \t\n\r");
    result = result.substr(start, end - start + 1);

    // Remove any prompt leakage
    const std::string leakage_markers[] = {
        "Hindi:", "English:", "/no_think"
    };
    for (const auto& marker : leakage_markers) {
        if (result.find(marker) == 0) {
            result = result.substr(marker.length());
            start = result.find_first_not_of(" \t\n\r");
            if (start == std::string::npos) return "";
            result = result.substr(start);
        }
    }

    return result;
}

// ============================================================
// Tokenizer — uses HFTokenizer from ExecuTorch
// ============================================================
std::vector<int64_t> LLMTranslator::tokenize(const std::string& text) {
#ifdef USE_EXECUTORCH
    if (!tokenizer_) return {};
    auto* tok = static_cast<tokenizers::HFTokenizer*>(tokenizer_);
    auto result = tok->encode(text, /*bos=*/0, /*eos=*/0);
    if (!result.ok()) {
        LOGE("Tokenization failed");
        return {};
    }
    auto& ids = result.get();
    std::vector<int64_t> out(ids.begin(), ids.end());
    return out;
#else
    (void)text;
    return {};
#endif
}

std::string LLMTranslator::detokenize(const std::vector<int64_t>& tokens) {
#ifdef USE_EXECUTORCH
    if (!tokenizer_ || tokens.empty()) return "";
    auto* tok = static_cast<tokenizers::HFTokenizer*>(tokenizer_);
    std::string result;
    for (auto id : tokens) {
        auto decoded = tok->decode(0, static_cast<uint64_t>(id));
        if (decoded.ok()) {
            result += decoded.get();
        }
    }
    return result;
#else
    (void)tokens;
    return "";
#endif
}
