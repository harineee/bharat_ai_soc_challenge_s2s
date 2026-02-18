/**
 * MT wrapper — three-mode translation routing:
 *   SPEED:    Marian NMT only (~49ms)
 *   BALANCED: NMT draft + LLM single-pass validation (~300-600ms)
 *   QUALITY:  NMT draft + LLM KV-cache speculative decode (~1-2s)
 */

#ifndef MT_WRAPPER_H
#define MT_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include "translation_mode.h"

#ifdef USE_ONNXRUNTIME
#include "marian_mt_wrapper.h"
#endif

#ifdef USE_EXECUTORCH
#include "llm_translator.h"
#endif

class MTWrapper {
public:
    MTWrapper();
    ~MTWrapper();

    // Initialize Marian NMT backend
    bool init(const std::string& model_path);

    // Initialize LLM backend (Qwen3-0.6B via ExecuTorch)
    bool init_llm(const std::string& llm_model_path,
                  const std::string& tokenizer_path);

    // Translate using the current translation mode
    std::string translate(const std::string& english_text);
    std::vector<std::string> translate_batch(const std::vector<std::string>& texts);

    // Translation mode control
    void set_translation_mode(TranslationMode mode);
    TranslationMode get_translation_mode() const { return mode_; }
    std::string get_mode_name() const;

    // Query backend availability
    bool is_llm_active() const;
    bool is_nmt_active() const;

    // Metrics (forwarded from LLM translator)
    double get_last_latency_ms() const;
    double get_acceptance_rate() const;
    int get_last_accepted_tokens() const;
    int get_last_total_draft_tokens() const;
    bool get_last_validation_confident() const;

private:
    // NMT-only translation (SPEED mode)
    std::string translate_nmt(const std::string& english_text);

    // Legacy full-prefill speculative (fallback)
    std::string translate_speculative_fullprefill(const std::string& english_text);

    // NMT + single-pass LLM validation (BALANCED mode)
    std::string translate_validated(const std::string& english_text);

    // NMT + KV-cache speculative decode (QUALITY mode)
    std::string translate_speculative_kv(const std::string& english_text);

    // Full LLM autoregressive (standalone, no NMT)
    std::string translate_llm(const std::string& english_text);

#ifdef USE_ONNXRUNTIME
    std::unique_ptr<MarianMTWrapper> marian_;
#endif
#ifdef USE_EXECUTORCH
    std::unique_ptr<LLMTranslator> llm_;
#endif
    bool use_placeholder_;
    bool use_llm_ = false;
    bool use_nmt_ = false;
    TranslationMode mode_ = TranslationMode::SPEED;
};

#endif // MT_WRAPPER_H
