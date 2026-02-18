/**
 * LLM-based translator using ExecuTorch + Qwen3-0.6B
 *
 * Supports four inference modes:
 * - AUTOREGRESSIVE: Standard token-by-token generation (full-prefill each step)
 * - SPECULATIVE: NMT draft verified token-by-token with full-prefill (legacy)
 * - SPECULATIVE_KV: NMT draft verified with KV-cache incremental decode (Quality)
 * - VALIDATE: Single-pass confidence check of NMT draft (Balanced)
 *
 * Uses XNNPACK backend with KleidiAI for SME2/NEON acceleration.
 */

#ifndef LLM_TRANSLATOR_H
#define LLM_TRANSLATOR_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

class LLMTranslator {
public:
    LLMTranslator();
    ~LLMTranslator();

    // Non-copyable
    LLMTranslator(const LLMTranslator&) = delete;
    LLMTranslator& operator=(const LLMTranslator&) = delete;

    /**
     * Initialize the LLM model and tokenizer.
     * @param model_path   Path to qwen3_0.6B.pte (ExecuTorch exported model)
     * @param tokenizer_path Path to tokenizer.json or tokenizer.model
     * @return true on success
     */
    bool init(const std::string& model_path,
              const std::string& tokenizer_path);

    /**
     * Full autoregressive translation (Quality mode).
     * Generates Hindi token-by-token from English prompt.
     * Slow (~375ms for 15 tokens) but highest quality.
     */
    std::string translate(const std::string& english_text);

    /**
     * Speculative translation using NMT draft (legacy, full-prefill per step).
     * O(n*seq_len) — slow on device (~6.5s). Kept as fallback.
     */
    std::string speculative_translate(const std::string& english_text,
                                      const std::string& nmt_draft);

    /**
     * KV-cache speculative decode (Quality mode).
     * Prefills prompt once, then verifies draft tokens one-by-one using
     * incremental KV-cache decode. On mismatch, continues autoregressive
     * from the rejection point. Cost: 1 prefill + N small decode calls.
     * Estimated ~1-2s on device (vs 6.5s full-prefill).
     */
    std::string speculative_translate_kv(const std::string& english_text,
                                          const std::string& nmt_draft);

    /**
     * Single-pass validation (Balanced mode).
     * Feeds prompt + entire draft in ONE forward call. Checks if the LLM
     * predicts EOS/punctuation after the draft (= high confidence) or a
     * content token (= lower confidence). Always returns the NMT draft
     * unchanged — this is a confidence check, not a rewrite.
     * Cost: 1 forward pass (~300-600ms on device).
     */
    std::string validate_translation(const std::string& english_text,
                                      const std::string& nmt_draft);

    bool is_initialized() const { return initialized_; }

    /// Did the last validate_translation() call indicate high confidence?
    bool get_last_validation_confident() const { return last_validation_confident_; }

    // Metrics (for performance tracking and report)
    double get_last_latency_ms() const { return last_latency_ms_; }
    int get_last_accepted_tokens() const { return last_accepted_tokens_; }
    int get_last_total_draft_tokens() const { return last_total_draft_tokens_; }
    double get_acceptance_rate() const;

private:
    // Prompt construction
    std::string build_prompt(const std::string& english_text);
    std::string build_refinement_prompt(const std::string& english_text,
                                        const std::string& nmt_draft);

    // Output cleaning
    std::string extract_translation(const std::string& raw_output);

    // Tokenizer operations
    std::vector<int64_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int64_t>& tokens);

    // Forward pass with optional repetition penalty.
    // Returns argmax token ID, or -1 on error.
    int64_t forward_next_token(const std::vector<int64_t>& token_ids,
                                int64_t start_pos,
                                const std::vector<int64_t>& penalize = {},
                                float rep_penalty = 1.0f);

    // ExecuTorch Module (opaque pointer, cast in .cpp)
    void* module_ = nullptr;

    // Tokenizer (opaque pointer, cast in .cpp)
    void* tokenizer_ = nullptr;

    bool initialized_ = false;
    bool last_validation_confident_ = false;

    // Metrics
    double last_latency_ms_ = 0.0;
    int last_accepted_tokens_ = 0;
    int last_total_draft_tokens_ = 0;
    int64_t total_accepted_ = 0;
    int64_t total_drafted_ = 0;

    std::string model_path_;
    std::string tokenizer_path_;
};

#endif // LLM_TRANSLATOR_H
