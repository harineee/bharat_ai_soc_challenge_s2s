/**
 * Marian MT wrapper using ONNX Runtime
 * Pure C++ — no Python dependencies.
 * Uses SentencePiece for tokenization/detokenization.
 */

#ifndef MARIAN_MT_WRAPPER_H
#define MARIAN_MT_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#endif

#ifdef USE_SENTENCEPIECE
#include <sentencepiece_processor.h>
#endif

class MarianMTWrapper {
public:
    MarianMTWrapper();
    ~MarianMTWrapper();

    bool init(const std::string& model_path);
    std::string translate(const std::string& english_text);

private:
#ifdef USE_ONNXRUNTIME
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::SessionOptions> session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;

    // Encoder
    std::unique_ptr<Ort::Session> encoder_session_;
    std::vector<const char*> encoder_input_names_;
    std::vector<const char*> encoder_output_names_;
    std::vector<std::string> encoder_input_names_s_;
    std::vector<std::string> encoder_output_names_s_;

    // Decoder (first step, no KV cache)
    std::unique_ptr<Ort::Session> decoder_session_;
    std::vector<const char*> decoder_input_names_;
    std::vector<const char*> decoder_output_names_;
    std::vector<std::string> decoder_input_names_s_;
    std::vector<std::string> decoder_output_names_s_;

    // Decoder with past (KV cache steps)
    std::unique_ptr<Ort::Session> decoder_past_session_;
    std::vector<const char*> decoder_past_input_names_;
    std::vector<const char*> decoder_past_output_names_;
    std::vector<std::string> decoder_past_input_names_s_;
    std::vector<std::string> decoder_past_output_names_s_;

    std::vector<int64_t> generate(const std::vector<int64_t>& input_tokens, int max_length = 50);
#endif

#ifdef USE_SENTENCEPIECE
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_source_;
    std::unique_ptr<sentencepiece::SentencePieceProcessor> sp_target_;
#endif

    // Tokenization (uses SentencePiece when available, fallback otherwise)
    std::vector<int64_t> tokenize(const std::string& text);
    std::string detokenize(const std::vector<int64_t>& tokens);

    // Fallback dictionary for when ONNX/SentencePiece unavailable
    std::unordered_map<std::string, std::string> basic_vocab_;
    void load_basic_vocab();

    // vocab.json mapping (token string → model ID, and reverse)
    std::unordered_map<std::string, int64_t> token_to_id_;
    std::unordered_map<int64_t, std::string> id_to_token_;
    bool load_vocab_json(const std::string& path);

    bool initialized_;
    std::string model_dir_;
};

#endif // MARIAN_MT_WRAPPER_H
