/**
 * Marian MT wrapper — pure C++ implementation.
 * Uses SentencePiece C++ library for tokenization.
 * Uses ONNX Runtime for encoder-decoder inference.
 * NO Python subprocess calls.
 */

#include "marian_mt_wrapper.h"
#include <iostream>
#include <algorithm>
#include <sstream>
#include <cctype>
#include <fstream>
#include <cmath>

MarianMTWrapper::MarianMTWrapper()
    : initialized_(false)
{
    load_basic_vocab();
}

MarianMTWrapper::~MarianMTWrapper() {
}

void MarianMTWrapper::load_basic_vocab() {
    basic_vocab_ = {
        {"hello", "नमस्ते"}, {"thank", "धन्यवाद"}, {"you", "आप"},
        {"yes", "हाँ"}, {"no", "नहीं"}, {"good", "अच्छा"},
        {"bad", "बुरा"}, {"water", "पानी"}, {"food", "भोजन"},
        {"help", "मदद"}, {"please", "कृपया"}, {"sorry", "माफ"},
        {"the", ""}, {"a", ""}, {"an", ""},
    };
}

// Minimal JSON parser for vocab.json ({"token": id, ...} format)
bool MarianMTWrapper::load_vocab_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Parse JSON object: iterate through "key": value pairs
    size_t pos = content.find('{');
    if (pos == std::string::npos) return false;
    pos++;

    while (pos < content.size()) {
        // Skip whitespace
        while (pos < content.size() && std::isspace(content[pos])) pos++;
        if (pos >= content.size() || content[pos] == '}') break;

        // Parse key (quoted string)
        if (content[pos] != '"') { pos++; continue; }
        pos++; // skip opening quote
        std::string key;
        while (pos < content.size() && content[pos] != '"') {
            if (content[pos] == '\\' && pos + 1 < content.size()) {
                pos++;
                if (content[pos] == 'u' && pos + 4 < content.size()) {
                    // Parse \uXXXX unicode escape
                    std::string hex = content.substr(pos + 1, 4);
                    unsigned int code = 0;
                    for (char c : hex) {
                        code <<= 4;
                        if (c >= '0' && c <= '9') code |= (c - '0');
                        else if (c >= 'a' && c <= 'f') code |= (c - 'a' + 10);
                        else if (c >= 'A' && c <= 'F') code |= (c - 'A' + 10);
                    }
                    // Convert to UTF-8
                    if (code < 0x80) {
                        key += static_cast<char>(code);
                    } else if (code < 0x800) {
                        key += static_cast<char>(0xC0 | (code >> 6));
                        key += static_cast<char>(0x80 | (code & 0x3F));
                    } else {
                        key += static_cast<char>(0xE0 | (code >> 12));
                        key += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                        key += static_cast<char>(0x80 | (code & 0x3F));
                    }
                    pos += 4;
                } else if (content[pos] == 'n') {
                    key += '\n';
                } else if (content[pos] == 't') {
                    key += '\t';
                } else if (content[pos] == '"') {
                    key += '"';
                } else if (content[pos] == '\\') {
                    key += '\\';
                } else {
                    key += content[pos];
                }
            } else {
                key += content[pos];
            }
            pos++;
        }
        if (pos < content.size()) pos++; // skip closing quote

        // Skip colon
        while (pos < content.size() && content[pos] != ':') pos++;
        if (pos < content.size()) pos++;

        // Skip whitespace
        while (pos < content.size() && std::isspace(content[pos])) pos++;

        // Parse value (integer)
        std::string num_str;
        while (pos < content.size() && (std::isdigit(content[pos]) || content[pos] == '-')) {
            num_str += content[pos];
            pos++;
        }

        if (!key.empty() && !num_str.empty()) {
            int64_t id = std::stoll(num_str);
            token_to_id_[key] = id;
            id_to_token_[id] = key;
        }

        // Skip comma
        while (pos < content.size() && content[pos] != ',' && content[pos] != '}') pos++;
        if (pos < content.size() && content[pos] == ',') pos++;
    }

    std::cout << "Loaded vocab.json: " << token_to_id_.size() << " tokens" << std::endl;
    return !token_to_id_.empty();
}

// ---------------------------------------------------------------------------
// Init: load ONNX sessions + SentencePiece models
// ---------------------------------------------------------------------------
bool MarianMTWrapper::init(const std::string& model_path) {
    // Extract model directory
    size_t last_slash = model_path.find_last_of("/\\");
    if (last_slash != std::string::npos) {
        model_dir_ = model_path.substr(0, last_slash + 1);
    } else {
        model_dir_ = "./";
    }

    // ---- Load vocab.json (token→ID mapping for ONNX model) ----
    if (!load_vocab_json(model_dir_ + "vocab.json")) {
        std::cerr << "Warning: vocab.json not loaded, will use SentencePiece IDs directly" << std::endl;
    }

    // ---- SentencePiece tokenizers ----
#ifdef USE_SENTENCEPIECE
    sp_source_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    sp_target_ = std::make_unique<sentencepiece::SentencePieceProcessor>();

    std::string src_spm = model_dir_ + "source.spm";
    std::string tgt_spm = model_dir_ + "target.spm";

    auto s1 = sp_source_->Load(src_spm);
    if (!s1.ok()) {
        std::cerr << "Failed to load source.spm: " << src_spm << std::endl;
        sp_source_.reset();
    } else {
        std::cout << "Loaded source tokenizer: " << src_spm
                  << " (vocab=" << sp_source_->GetPieceSize() << ")" << std::endl;
    }

    auto s2 = sp_target_->Load(tgt_spm);
    if (!s2.ok()) {
        std::cerr << "Failed to load target.spm: " << tgt_spm << std::endl;
        sp_target_.reset();
    } else {
        std::cout << "Loaded target tokenizer: " << tgt_spm
                  << " (vocab=" << sp_target_->GetPieceSize() << ")" << std::endl;
    }
#endif

    // ---- ONNX Runtime sessions ----
#ifdef USE_ONNXRUNTIME
    try {
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "MarianMT");
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetIntraOpNumThreads(2);  // Mobile-friendly
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        auto load_session = [&](const std::string& path,
                                std::unique_ptr<Ort::Session>& session,
                                std::vector<std::string>& in_s,
                                std::vector<std::string>& out_s,
                                std::vector<const char*>& in_c,
                                std::vector<const char*>& out_c) {
            session = std::make_unique<Ort::Session>(*env_, path.c_str(), *session_options_);
            for (size_t i = 0; i < session->GetInputCount(); i++) {
                in_s.push_back(session->GetInputNameAllocated(i, allocator_).get());
            }
            for (size_t i = 0; i < session->GetOutputCount(); i++) {
                out_s.push_back(session->GetOutputNameAllocated(i, allocator_).get());
            }
            for (auto& n : in_s)  in_c.push_back(n.c_str());
            for (auto& n : out_s) out_c.push_back(n.c_str());
        };

        load_session(model_dir_ + "encoder_model.onnx",
                     encoder_session_,
                     encoder_input_names_s_, encoder_output_names_s_,
                     encoder_input_names_, encoder_output_names_);

        load_session(model_dir_ + "decoder_model.onnx",
                     decoder_session_,
                     decoder_input_names_s_, decoder_output_names_s_,
                     decoder_input_names_, decoder_output_names_);

        load_session(model_dir_ + "decoder_with_past_model.onnx",
                     decoder_past_session_,
                     decoder_past_input_names_s_, decoder_past_output_names_s_,
                     decoder_past_input_names_, decoder_past_output_names_);

        initialized_ = true;
        std::cout << "Marian MT (ONNX) initialized" << std::endl;
        return true;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime init failed: " << e.what() << std::endl;
        initialized_ = false;
        return true;  // fall through to basic vocab
    }
#else
    std::cerr << "ONNX Runtime not available, using basic word translation" << std::endl;
    (void)model_path;
    return true;
#endif
}

// ---------------------------------------------------------------------------
// Tokenize — uses SentencePiece source.spm
// ---------------------------------------------------------------------------
std::vector<int64_t> MarianMTWrapper::tokenize(const std::string& text) {
    std::vector<int64_t> ids;

#ifdef USE_SENTENCEPIECE
    if (sp_source_) {
        // Use SentencePiece for segmentation, then map to vocab.json IDs
        std::vector<std::string> pieces;
        auto status = sp_source_->Encode(text, &pieces);
        if (status.ok() && !pieces.empty()) {
            if (!token_to_id_.empty()) {
                // Map token strings → vocab.json IDs
                for (const auto& piece : pieces) {
                    auto it = token_to_id_.find(piece);
                    if (it != token_to_id_.end()) {
                        ids.push_back(it->second);
                    } else {
                        // Try <unk> token
                        auto unk = token_to_id_.find("<unk>");
                        if (unk != token_to_id_.end()) {
                            ids.push_back(unk->second);
                        }
                    }
                }
            } else {
                // Fallback: use SentencePiece IDs directly (less accurate)
                std::vector<int> sp_ids;
                sp_source_->Encode(text, &sp_ids);
                for (int id : sp_ids) {
                    ids.push_back(static_cast<int64_t>(id));
                }
            }
            ids.push_back(0);  // EOS token (</s> = 0)
            return ids;
        }
    }
#endif

    // Fallback: hash-based (poor quality, for build testing only)
    std::istringstream iss(text);
    std::string word;
    while (iss >> word) {
        std::transform(word.begin(), word.end(), word.begin(), ::tolower);
        word.erase(std::remove_if(word.begin(), word.end(), ::ispunct), word.end());
        if (!word.empty()) {
            ids.push_back(static_cast<int64_t>((std::hash<std::string>{}(word) % 50000) + 3));
        }
    }
    ids.push_back(0);
    return ids;
}

// ---------------------------------------------------------------------------
// Detokenize — uses SentencePiece target.spm
// ---------------------------------------------------------------------------
std::string MarianMTWrapper::detokenize(const std::vector<int64_t>& tokens) {
    // If we have vocab.json mapping, use it for decoding
    if (!id_to_token_.empty()) {
        std::string result;
        for (auto t : tokens) {
            if (t == 0 || t == 61949) continue;  // Skip EOS and PAD
            auto it = id_to_token_.find(t);
            if (it != id_to_token_.end()) {
                result += it->second;
            }
        }
        // Clean SentencePiece format: replace ▁ (U+2581) with space
        std::string cleaned;
        for (size_t i = 0; i < result.size(); ) {
            // Check for ▁ (UTF-8: 0xE2 0x96 0x81)
            if (i + 2 < result.size() &&
                (unsigned char)result[i] == 0xE2 &&
                (unsigned char)result[i+1] == 0x96 &&
                (unsigned char)result[i+2] == 0x81) {
                if (!cleaned.empty()) cleaned += ' ';
                i += 3;
            } else {
                cleaned += result[i];
                i++;
            }
        }
        if (!cleaned.empty()) return cleaned;
    }

#ifdef USE_SENTENCEPIECE
    if (sp_target_) {
        std::vector<int> int_ids;
        for (auto t : tokens) {
            if (t == 0 || t == 61949) continue;  // Skip EOS and PAD
            int_ids.push_back(static_cast<int>(t));
        }
        std::string result;
        auto status = sp_target_->Decode(int_ids, &result);
        if (status.ok() && !result.empty()) {
            return result;
        }
    }
#endif

    // Fallback: show raw IDs
    std::ostringstream oss;
    bool first = true;
    for (auto t : tokens) {
        if (t == 0 || t == 61949) continue;
        if (!first) oss << " ";
        oss << "[" << t << "]";
        first = false;
    }
    return oss.str();
}

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------
std::string MarianMTWrapper::translate(const std::string& english_text) {
    if (english_text.empty()) return "";

#ifdef USE_ONNXRUNTIME
    if (initialized_ && encoder_session_) {
        try {
            auto input_tokens = tokenize(english_text);
            auto output_tokens = generate(input_tokens, 50);
            std::string result = detokenize(output_tokens);
            if (!result.empty()) return result;
        } catch (const Ort::Exception& e) {
            std::cerr << "ONNX inference error: " << e.what() << std::endl;
        }
    }
#endif

    // Fallback word-by-word
    std::istringstream iss(english_text);
    std::ostringstream oss;
    std::string word;
    bool first = true;
    while (iss >> word) {
        std::string clean = word;
        std::transform(clean.begin(), clean.end(), clean.begin(), ::tolower);
        clean.erase(std::remove_if(clean.begin(), clean.end(), ::ispunct), clean.end());
        auto it = basic_vocab_.find(clean);
        if (it != basic_vocab_.end() && !it->second.empty()) {
            if (!first) oss << " ";
            oss << it->second;
            first = false;
        } else if (!clean.empty()) {
            if (!first) oss << " ";
            oss << word;
            first = false;
        }
    }
    std::string r = oss.str();
    return r.empty() ? ("[HI: " + english_text + "]") : r;
}

// ---------------------------------------------------------------------------
// Autoregressive generation (encoder-decoder)
// Uses decoder_model for first step, decoder_with_past for subsequent steps.
// ---------------------------------------------------------------------------
#ifdef USE_ONNXRUNTIME
std::vector<int64_t> MarianMTWrapper::generate(
    const std::vector<int64_t>& input_tokens, int max_length) {

    auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ---- Encoder ----
    std::vector<int64_t> input_shape = {1, (int64_t)input_tokens.size()};
    std::vector<int64_t> ids_data = input_tokens;
    std::vector<int64_t> attn_data(input_tokens.size(), 1);

    Ort::Value enc_ids = Ort::Value::CreateTensor<int64_t>(
        mem, ids_data.data(), ids_data.size(), input_shape.data(), 2);
    Ort::Value enc_attn = Ort::Value::CreateTensor<int64_t>(
        mem, attn_data.data(), attn_data.size(), input_shape.data(), 2);

    std::vector<Ort::Value> enc_in;
    enc_in.push_back(std::move(enc_ids));
    enc_in.push_back(std::move(enc_attn));

    auto enc_out = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        encoder_input_names_.data(), enc_in.data(), enc_in.size(),
        encoder_output_names_.data(), encoder_output_names_.size());

    auto& enc_hidden = enc_out[0];

    // ---- Decoder: first step ----
    std::vector<int64_t> generated;
    int64_t bos = 61949;  // decoder_start_token_id
    int effective_max = std::min(max_length, (int)(input_tokens.size() * 3 + 10));

    // First step uses decoder_model.onnx (no past)
    {
        std::vector<int64_t> dec_ids = {bos};
        std::vector<int64_t> dec_shape = {1, 1};

        std::vector<int64_t> enc_attn_d(input_tokens.size(), 1);
        Ort::Value d_attn = Ort::Value::CreateTensor<int64_t>(
            mem, enc_attn_d.data(), enc_attn_d.size(), input_shape.data(), 2);
        Ort::Value d_ids = Ort::Value::CreateTensor<int64_t>(
            mem, dec_ids.data(), dec_ids.size(), dec_shape.data(), 2);

        auto enc_h_data = enc_hidden.GetTensorMutableData<float>();
        auto enc_h_shape = enc_hidden.GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value d_enc = Ort::Value::CreateTensor<float>(
            mem, enc_h_data,
            enc_hidden.GetTensorTypeAndShapeInfo().GetElementCount(),
            enc_h_shape.data(), enc_h_shape.size());

        std::vector<Ort::Value> dec_in;
        dec_in.push_back(std::move(d_attn));
        dec_in.push_back(std::move(d_ids));
        dec_in.push_back(std::move(d_enc));

        auto dec_out = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            decoder_input_names_.data(), dec_in.data(), dec_in.size(),
            decoder_output_names_.data(), decoder_output_names_.size());

        // Extract logits [1, 1, vocab_size]
        float* logits = dec_out[0].GetTensorMutableData<float>();
        auto lshape = dec_out[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab = lshape[2];

        int64_t next = 0;
        float mx = -1e30f;
        for (int64_t i = 0; i < vocab; i++) {
            if (i == 61949) continue;  // skip pad
            if (logits[i] > mx) { mx = logits[i]; next = i; }
        }
        if (next == 0) return generated;  // EOS immediately
        generated.push_back(next);

        // Try to use KV cache from decoder_with_past for remaining steps
        if (decoder_past_session_ && dec_out.size() > 1) {
            // dec_out layout (25 outputs from decoder_model.onnx):
            //   [0] logits
            //   [1+i*4+0] present.i.decoder.key
            //   [1+i*4+1] present.i.decoder.value
            //   [1+i*4+2] present.i.encoder.key  (static)
            //   [1+i*4+3] present.i.encoder.value (static)
            //
            // decoder_with_past expects 26 inputs:
            //   [0] encoder_attention_mask
            //   [1] input_ids
            //   [2+i*4+0..3] past_key_values.i.{decoder.key, decoder.value, encoder.key, encoder.value}
            //
            // decoder_with_past outputs 13:
            //   [0] logits
            //   [1+i*2+0] present.i.decoder.key (updated)
            //   [1+i*2+1] present.i.decoder.value (updated)
            //   (encoder KV not output — it's static)

            int num_layers = (int)(dec_out.size() - 1) / 4;  // 6

            // Save encoder KV tensor data (static throughout generation)
            struct TensorRef { float* data; std::vector<int64_t> shape; size_t count; };
            std::vector<TensorRef> enc_kv(num_layers * 2);
            for (int i = 0; i < num_layers; i++) {
                int ek = 1 + i * 4 + 2;
                int ev = 1 + i * 4 + 3;
                enc_kv[i*2]   = {dec_out[ek].GetTensorMutableData<float>(),
                                 dec_out[ek].GetTensorTypeAndShapeInfo().GetShape(),
                                 dec_out[ek].GetTensorTypeAndShapeInfo().GetElementCount()};
                enc_kv[i*2+1] = {dec_out[ev].GetTensorMutableData<float>(),
                                 dec_out[ev].GetTensorTypeAndShapeInfo().GetShape(),
                                 dec_out[ev].GetTensorTypeAndShapeInfo().GetElementCount()};
            }

            // Keep past outputs alive so tensor pointers remain valid
            std::vector<Ort::Value> latest_past_out;
            bool first_kv_step = true;

            for (int step = 1; step < effective_max; step++) {
                try {
                    std::vector<Ort::Value> past_in;

                    // [0] encoder_attention_mask
                    std::vector<int64_t> ea(input_tokens.size(), 1);
                    past_in.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem, ea.data(), ea.size(), input_shape.data(), 2));

                    // [1] input_ids: last generated token
                    std::vector<int64_t> last_tok = {generated.back()};
                    std::vector<int64_t> one_shape = {1, 1};
                    past_in.push_back(Ort::Value::CreateTensor<int64_t>(
                        mem, last_tok.data(), 1, one_shape.data(), 2));

                    // [2..25] past_key_values: interleave decoder KV + encoder KV
                    for (int i = 0; i < num_layers; i++) {
                        if (first_kv_step) {
                            // Decoder KV from initial decoder_model output
                            int dk = 1 + i * 4 + 0;
                            int dv = 1 + i * 4 + 1;
                            auto dk_shape = dec_out[dk].GetTensorTypeAndShapeInfo().GetShape();
                            auto dv_shape = dec_out[dv].GetTensorTypeAndShapeInfo().GetShape();
                            past_in.push_back(Ort::Value::CreateTensor<float>(
                                mem, dec_out[dk].GetTensorMutableData<float>(),
                                dec_out[dk].GetTensorTypeAndShapeInfo().GetElementCount(),
                                dk_shape.data(), dk_shape.size()));
                            past_in.push_back(Ort::Value::CreateTensor<float>(
                                mem, dec_out[dv].GetTensorMutableData<float>(),
                                dec_out[dv].GetTensorTypeAndShapeInfo().GetElementCount(),
                                dv_shape.data(), dv_shape.size()));
                        } else {
                            // Decoder KV from latest decoder_with_past output
                            int dk = 1 + i * 2;
                            int dv = 1 + i * 2 + 1;
                            auto dk_shape = latest_past_out[dk].GetTensorTypeAndShapeInfo().GetShape();
                            auto dv_shape = latest_past_out[dv].GetTensorTypeAndShapeInfo().GetShape();
                            past_in.push_back(Ort::Value::CreateTensor<float>(
                                mem, latest_past_out[dk].GetTensorMutableData<float>(),
                                latest_past_out[dk].GetTensorTypeAndShapeInfo().GetElementCount(),
                                dk_shape.data(), dk_shape.size()));
                            past_in.push_back(Ort::Value::CreateTensor<float>(
                                mem, latest_past_out[dv].GetTensorMutableData<float>(),
                                latest_past_out[dv].GetTensorTypeAndShapeInfo().GetElementCount(),
                                dv_shape.data(), dv_shape.size()));
                        }
                        // Encoder KV (static from first decoder step)
                        past_in.push_back(Ort::Value::CreateTensor<float>(
                            mem, enc_kv[i*2].data, enc_kv[i*2].count,
                            enc_kv[i*2].shape.data(), enc_kv[i*2].shape.size()));
                        past_in.push_back(Ort::Value::CreateTensor<float>(
                            mem, enc_kv[i*2+1].data, enc_kv[i*2+1].count,
                            enc_kv[i*2+1].shape.data(), enc_kv[i*2+1].shape.size()));
                    }

                    if (past_in.size() != decoder_past_input_names_.size()) {
                        break;
                    }

                    auto past_out = decoder_past_session_->Run(
                        Ort::RunOptions{nullptr},
                        decoder_past_input_names_.data(), past_in.data(), past_in.size(),
                        decoder_past_output_names_.data(), decoder_past_output_names_.size());

                    float* pl = past_out[0].GetTensorMutableData<float>();
                    auto ps = past_out[0].GetTensorTypeAndShapeInfo().GetShape();
                    int64_t pv = ps[2];

                    // Repetition penalty
                    for (int64_t prev : generated) {
                        if (prev >= 0 && prev < pv) {
                            pl[prev] = (pl[prev] > 0) ? pl[prev] / 1.5f : pl[prev] * 1.5f;
                        }
                    }

                    int64_t nt = 0; float nmx = -1e30f;
                    for (int64_t i = 0; i < pv; i++) {
                        if (i == 61949) continue;
                        if (pl[i] > nmx) { nmx = pl[i]; nt = i; }
                    }
                    if (nt == 0) break;  // EOS
                    generated.push_back(nt);

                    latest_past_out = std::move(past_out);
                    first_kv_step = false;

                } catch (const Ort::Exception& e) {
                    std::cerr << "KV-cache step failed: " << e.what() << std::endl;
                    break;
                }
            }
            return generated;
        }
    }

    // ---- Fallback: non-cached decoder loop (O(n^2) but always works) ----
    for (int step = 1; step < effective_max; step++) {
        std::vector<int64_t> full_dec = {bos};
        full_dec.insert(full_dec.end(), generated.begin(), generated.end());
        std::vector<int64_t> dshape = {1, (int64_t)full_dec.size()};

        std::vector<int64_t> ea(input_tokens.size(), 1);
        Ort::Value d_attn = Ort::Value::CreateTensor<int64_t>(
            mem, ea.data(), ea.size(), input_shape.data(), 2);
        Ort::Value d_ids = Ort::Value::CreateTensor<int64_t>(
            mem, full_dec.data(), full_dec.size(), dshape.data(), 2);

        auto eh = enc_hidden.GetTensorMutableData<float>();
        auto ehs = enc_hidden.GetTensorTypeAndShapeInfo().GetShape();
        Ort::Value d_enc = Ort::Value::CreateTensor<float>(
            mem, eh,
            enc_hidden.GetTensorTypeAndShapeInfo().GetElementCount(),
            ehs.data(), ehs.size());

        std::vector<Ort::Value> di;
        di.push_back(std::move(d_attn));
        di.push_back(std::move(d_ids));
        di.push_back(std::move(d_enc));

        auto dout = decoder_session_->Run(
            Ort::RunOptions{nullptr},
            decoder_input_names_.data(), di.data(), di.size(),
            decoder_output_names_.data(), 1);

        float* logits = dout[0].GetTensorMutableData<float>();
        auto ls = dout[0].GetTensorTypeAndShapeInfo().GetShape();
        int64_t vocab = ls[2], seq = ls[1];
        float* last = logits + (seq - 1) * vocab;

        for (int64_t prev : generated) {
            if (prev >= 0 && prev < vocab) {
                last[prev] = (last[prev] > 0) ? last[prev] / 1.5f : last[prev] * 1.5f;
            }
        }

        int64_t nt = 0; float nmx = -1e30f;
        for (int64_t i = 0; i < vocab; i++) {
            if (i == 61949) continue;
            if (last[i] > nmx) { nmx = last[i]; nt = i; }
        }
        if (nt == 0) break;
        generated.push_back(nt);
    }
    return generated;
}
#endif
