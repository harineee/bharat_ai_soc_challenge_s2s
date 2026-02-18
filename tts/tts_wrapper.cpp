/**
 * TTS wrapper — native C++ Piper VITS inference via ONNX Runtime.
 * Includes Devanagari → IPA phonemization + linear resampling.
 * NO Python. NO popen().
 */

#include "tts_wrapper.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstring>

#ifdef __ANDROID__
#include <android/log.h>
#define TTS_LOG(...) __android_log_print(ANDROID_LOG_INFO, "TTS", __VA_ARGS__)
#else
#define TTS_LOG(...) do { fprintf(stderr, "[TTS] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

// -----------------------------------------------------------------------
// Devanagari → IPA table (covers standard Hindi consonants + vowels)
// -----------------------------------------------------------------------
void TTSWrapper::init_devanagari_table() {
    // Vowels (independent forms)
    devanagari_to_ipa_["अ"] = "ə";
    devanagari_to_ipa_["आ"] = "aː";
    devanagari_to_ipa_["इ"] = "ɪ";
    devanagari_to_ipa_["ई"] = "iː";
    devanagari_to_ipa_["उ"] = "ʊ";
    devanagari_to_ipa_["ऊ"] = "uː";
    devanagari_to_ipa_["ऋ"] = "ɾɪ";
    devanagari_to_ipa_["ए"] = "eː";
    devanagari_to_ipa_["ऐ"] = "ɛː";
    devanagari_to_ipa_["ओ"] = "oː";
    devanagari_to_ipa_["औ"] = "ɔː";

    // Vowel matras (dependent forms)
    devanagari_to_ipa_["ा"] = "aː";
    devanagari_to_ipa_["ि"] = "ɪ";
    devanagari_to_ipa_["ी"] = "iː";
    devanagari_to_ipa_["ु"] = "ʊ";
    devanagari_to_ipa_["ू"] = "uː";
    devanagari_to_ipa_["ृ"] = "ɾɪ";
    devanagari_to_ipa_["े"] = "eː";
    devanagari_to_ipa_["ै"] = "ɛː";
    devanagari_to_ipa_["ो"] = "oː";
    devanagari_to_ipa_["ौ"] = "ɔː";

    // Consonants (velar)
    devanagari_to_ipa_["क"] = "kə";
    devanagari_to_ipa_["ख"] = "kʰə";
    devanagari_to_ipa_["ग"] = "ɡə";
    devanagari_to_ipa_["घ"] = "ɡʰə";
    devanagari_to_ipa_["ङ"] = "ŋə";

    // Consonants (palatal)
    devanagari_to_ipa_["च"] = "t͡ʃə";
    devanagari_to_ipa_["छ"] = "t͡ʃʰə";
    devanagari_to_ipa_["ज"] = "d͡ʒə";
    devanagari_to_ipa_["झ"] = "d͡ʒʰə";
    devanagari_to_ipa_["ञ"] = "ɲə";

    // Consonants (retroflex)
    devanagari_to_ipa_["ट"] = "ʈə";
    devanagari_to_ipa_["ठ"] = "ʈʰə";
    devanagari_to_ipa_["ड"] = "ɖə";
    devanagari_to_ipa_["ढ"] = "ɖʰə";
    devanagari_to_ipa_["ण"] = "ɳə";

    // Consonants (dental)
    devanagari_to_ipa_["त"] = "t̪ə";
    devanagari_to_ipa_["थ"] = "t̪ʰə";
    devanagari_to_ipa_["द"] = "d̪ə";
    devanagari_to_ipa_["ध"] = "d̪ʰə";
    devanagari_to_ipa_["न"] = "nə";

    // Consonants (labial)
    devanagari_to_ipa_["प"] = "pə";
    devanagari_to_ipa_["फ"] = "pʰə";
    devanagari_to_ipa_["ब"] = "bə";
    devanagari_to_ipa_["भ"] = "bʰə";
    devanagari_to_ipa_["म"] = "mə";

    // Semi-vowels / approximants
    devanagari_to_ipa_["य"] = "jə";
    devanagari_to_ipa_["र"] = "ɾə";
    devanagari_to_ipa_["ल"] = "lə";
    devanagari_to_ipa_["व"] = "ʋə";

    // Sibilants / fricatives
    devanagari_to_ipa_["श"] = "ʃə";
    devanagari_to_ipa_["ष"] = "ʂə";
    devanagari_to_ipa_["स"] = "sə";
    devanagari_to_ipa_["ह"] = "ɦə";

    // Nukta variants
    devanagari_to_ipa_["ड़"] = "ɽə";
    devanagari_to_ipa_["ढ़"] = "ɽʰə";
    devanagari_to_ipa_["क़"] = "qə";
    devanagari_to_ipa_["ख़"] = "xə";
    devanagari_to_ipa_["ग़"] = "ɣə";
    devanagari_to_ipa_["ज़"] = "zə";
    devanagari_to_ipa_["फ़"] = "fə";

    // Virama (halant) — strips inherent vowel
    devanagari_to_ipa_["्"] = "";

    // Anusvara, Visarga, Chandrabindu
    devanagari_to_ipa_["ं"] = "̃";   // nasalization
    devanagari_to_ipa_["ः"] = "h";
    devanagari_to_ipa_["ँ"] = "̃";

    // Devanagari digits → IPA (just map to digit phonemes)
    devanagari_to_ipa_["०"] = "0"; devanagari_to_ipa_["१"] = "1";
    devanagari_to_ipa_["२"] = "2"; devanagari_to_ipa_["३"] = "3";
    devanagari_to_ipa_["४"] = "4"; devanagari_to_ipa_["५"] = "5";
    devanagari_to_ipa_["६"] = "6"; devanagari_to_ipa_["७"] = "7";
    devanagari_to_ipa_["८"] = "8"; devanagari_to_ipa_["९"] = "9";

    // Punctuation
    devanagari_to_ipa_["।"] = ".";
    devanagari_to_ipa_["॥"] = ".";
}

// -----------------------------------------------------------------------
// Parse Piper JSON config (minimal JSON parser for known structure)
// -----------------------------------------------------------------------
bool TTSWrapper::load_config(const std::string& json_path) {
    std::ifstream f(json_path);
    if (!f.is_open()) return false;

    std::string content((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());
    f.close();

    // Extract sample_rate
    auto extract_int = [&](const std::string& key) -> int {
        size_t p = content.find("\"" + key + "\"");
        if (p == std::string::npos) return -1;
        p = content.find(":", p);
        if (p == std::string::npos) return -1;
        p = content.find_first_of("0123456789", p);
        if (p == std::string::npos) return -1;
        return std::stoi(content.substr(p));
    };

    auto extract_float = [&](const std::string& key) -> float {
        size_t p = content.find("\"" + key + "\"");
        if (p == std::string::npos) return -1.0f;
        p = content.find(":", p);
        if (p == std::string::npos) return -1.0f;
        p = content.find_first_of("0123456789.", p);
        if (p == std::string::npos) return -1.0f;
        return std::stof(content.substr(p));
    };

    int sr = extract_int("sample_rate");
    if (sr > 0) model_sample_rate_ = sr;

    float ns = extract_float("noise_scale");
    if (ns >= 0) noise_scale_ = ns;

    float ls = extract_float("length_scale");
    if (ls >= 0) length_scale_ = ls;

    float nw = extract_float("noise_w");
    if (nw >= 0) noise_w_ = nw;

    // Parse phoneme_id_map
    size_t map_start = content.find("\"phoneme_id_map\"");
    if (map_start != std::string::npos) {
        size_t brace = content.find("{", map_start + 16);
        if (brace == std::string::npos) return true;

        size_t pos = brace + 1;
        int depth = 1;
        while (pos < content.size() && depth > 0) {
            // Find next key
            size_t q1 = content.find("\"", pos);
            if (q1 == std::string::npos) break;
            size_t q2 = content.find("\"", q1 + 1);
            if (q2 == std::string::npos) break;

            std::string key = content.substr(q1 + 1, q2 - q1 - 1);

            // Find array [id, ...]
            size_t arr_start = content.find("[", q2);
            size_t arr_end = content.find("]", arr_start);
            if (arr_start == std::string::npos || arr_end == std::string::npos) break;

            std::vector<int64_t> ids;
            std::string arr_content = content.substr(arr_start + 1, arr_end - arr_start - 1);
            std::istringstream iss(arr_content);
            std::string num;
            while (std::getline(iss, num, ',')) {
                try {
                    ids.push_back(std::stoll(num));
                } catch (...) {}
            }

            if (!key.empty() && !ids.empty()) {
                phoneme_id_map_[key] = ids;
            }

            pos = arr_end + 1;
            // Check for end of map
            size_t next_q = content.find("\"", pos);
            size_t next_b = content.find("}", pos);
            if (next_b != std::string::npos && (next_q == std::string::npos || next_b < next_q)) {
                break;  // End of phoneme_id_map
            }
        }
    }

    std::cout << "TTS config: sample_rate=" << model_sample_rate_
              << " noise_scale=" << noise_scale_
              << " length_scale=" << length_scale_
              << " phonemes=" << phoneme_id_map_.size() << std::endl;
    return true;
}

// -----------------------------------------------------------------------
// Convert IPA string → phoneme IDs using phoneme_id_map
// Piper VITS requires interspersed pad tokens (id 0) between every phoneme.
// Output format: [0, BOS, 0, p1, 0, p2, 0, ..., 0, pN, 0, EOS, 0]
// -----------------------------------------------------------------------
std::vector<int64_t> TTSWrapper::ipa_to_ids(const std::string& ipa) {
    // First collect raw phoneme IDs without padding
    std::vector<int64_t> raw_ids;

    // BOS
    auto it_bos = phoneme_id_map_.find("^");
    if (it_bos != phoneme_id_map_.end()) {
        raw_ids.insert(raw_ids.end(), it_bos->second.begin(), it_bos->second.end());
    }

    // Process IPA characters
    size_t i = 0;
    int matched = 0, skipped = 0;
    while (i < ipa.size()) {
        bool found = false;

        // Try multi-byte UTF-8 sequences (up to 4 bytes)
        for (int len = 4; len >= 1; len--) {
            if (i + len > ipa.size()) continue;
            std::string sub = ipa.substr(i, len);
            auto it = phoneme_id_map_.find(sub);
            if (it != phoneme_id_map_.end()) {
                raw_ids.insert(raw_ids.end(), it->second.begin(), it->second.end());
                i += len;
                found = true;
                matched++;
                break;
            }
        }

        if (!found) {
            skipped++;
            // Advance by one UTF-8 character
            unsigned char c = ipa[i];
            int skip = 1;
            if ((c & 0xE0) == 0xC0) skip = 2;
            else if ((c & 0xF0) == 0xE0) skip = 3;
            else if ((c & 0xF8) == 0xF0) skip = 4;
            i += skip;
        }
    }

    // EOS
    auto it_eos = phoneme_id_map_.find("$");
    if (it_eos != phoneme_id_map_.end()) {
        raw_ids.insert(raw_ids.end(), it_eos->second.begin(), it_eos->second.end());
    }

    TTS_LOG("ipa_to_ids: %d matched, %d skipped, %zu raw IDs", matched, skipped, raw_ids.size());

    // Intersperse pad tokens (id 0) — required by Piper VITS
    // Pattern: [0, id1, 0, id2, 0, ..., 0, idN, 0]
    std::vector<int64_t> ids;
    ids.reserve(raw_ids.size() * 2 + 1);
    for (size_t j = 0; j < raw_ids.size(); j++) {
        ids.push_back(0);  // pad
        ids.push_back(raw_ids[j]);
    }
    ids.push_back(0);  // trailing pad

    return ids;
}

// -----------------------------------------------------------------------
// Hindi text → phoneme IDs
// -----------------------------------------------------------------------
std::vector<int64_t> TTSWrapper::text_to_phoneme_ids(const std::string& text) {
    std::string ipa;
    bool prev_was_consonant = false;

    size_t i = 0;
    while (i < text.size()) {
        unsigned char c = text[i];
        int char_len = 1;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        if (i + char_len > text.size()) break;
        std::string ch = text.substr(i, char_len);

        // Check for multi-character sequences first (nukta variants: 2 codepoints)
        bool found_multi = false;
        if (i + char_len < text.size()) {
            int next_len = 1;
            unsigned char nc = text[i + char_len];
            if ((nc & 0xE0) == 0xC0) next_len = 2;
            else if ((nc & 0xF0) == 0xE0) next_len = 3;
            else if ((nc & 0xF8) == 0xF0) next_len = 4;

            if (i + char_len + next_len <= text.size()) {
                std::string duo = text.substr(i, char_len + next_len);
                auto it = devanagari_to_ipa_.find(duo);
                if (it != devanagari_to_ipa_.end()) {
                    // Check if it's virama — strip the inherent vowel from prev consonant
                    if (it->second.empty() && prev_was_consonant && ipa.size() >= 2) {
                        // Remove trailing "ə" from previous consonant
                        if (ipa.size() >= 2 && ipa.substr(ipa.size() - 2) == "ə") {
                            ipa.erase(ipa.size() - 2);
                        }
                    } else {
                        if (prev_was_consonant && !it->second.empty()) {
                            // Matra after consonant: strip inherent vowel
                            if (ipa.size() >= 2 && ipa.substr(ipa.size() - 2) == "ə") {
                                ipa.erase(ipa.size() - 2);
                            }
                        }
                        ipa += it->second;
                    }
                    prev_was_consonant = false;
                    i += char_len + next_len;
                    found_multi = true;
                }
            }
        }

        if (!found_multi) {
            auto it = devanagari_to_ipa_.find(ch);
            if (it != devanagari_to_ipa_.end()) {
                if (it->second.empty()) {
                    // Virama: strip inherent vowel from previous consonant
                    if (prev_was_consonant && ipa.size() >= 2 &&
                        ipa.substr(ipa.size() - 2) == "ə") {
                        ipa.erase(ipa.size() - 2);
                    }
                    prev_was_consonant = false;
                } else {
                    // Check if this is a matra (dependent vowel)
                    bool is_matra = (ch == "ा" || ch == "ि" || ch == "ी" ||
                                     ch == "ु" || ch == "ू" || ch == "ृ" ||
                                     ch == "े" || ch == "ै" || ch == "ो" ||
                                     ch == "ौ");
                    if (is_matra && prev_was_consonant) {
                        // Remove inherent "ə" before adding matra vowel
                        if (ipa.size() >= 2 && ipa.substr(ipa.size() - 2) == "ə") {
                            ipa.erase(ipa.size() - 2);
                        }
                    }

                    ipa += it->second;

                    // Check if this char maps to a consonant (contains "ə" at end)
                    std::string mapped = it->second;
                    prev_was_consonant = (mapped.size() >= 2 &&
                                          mapped.substr(mapped.size() - 2) == "ə");
                }
            } else {
                // ASCII / spaces / punctuation — pass through
                if (ch == " ") {
                    ipa += " ";
                    prev_was_consonant = false;
                } else if (ch.size() == 1 && std::isprint(ch[0])) {
                    // Map printable ASCII directly (punctuation, digits)
                    ipa += ch;
                    prev_was_consonant = false;
                }
                // Skip other unknown chars
            }
            i += char_len;
        }
    }

    TTS_LOG("Devanagari→IPA: \"%s\"", ipa.c_str());
    return ipa_to_ids(ipa);
}

// -----------------------------------------------------------------------
// Linear resampling (e.g. 22050 → 16000)
// -----------------------------------------------------------------------
std::vector<float> TTSWrapper::resample(const std::vector<float>& audio,
                                         int from_rate, int to_rate) {
    if (from_rate == to_rate || audio.empty()) return audio;

    double ratio = static_cast<double>(to_rate) / from_rate;
    size_t out_len = static_cast<size_t>(audio.size() * ratio);
    std::vector<float> output(out_len);

    for (size_t i = 0; i < out_len; i++) {
        double src = i / ratio;
        size_t i0 = static_cast<size_t>(src);
        size_t i1 = std::min(i0 + 1, audio.size() - 1);
        double frac = src - i0;
        output[i] = static_cast<float>(audio[i0] * (1.0 - frac) + audio[i1] * frac);
    }
    return output;
}

// -----------------------------------------------------------------------
// Constructor / Destructor
// -----------------------------------------------------------------------
TTSWrapper::TTSWrapper() {
    init_devanagari_table();
}

TTSWrapper::~TTSWrapper() {
#ifdef USE_ONNXRUNTIME
    delete session_; session_ = nullptr;
    delete env_; env_ = nullptr;
#endif
}

// -----------------------------------------------------------------------
// Init
// -----------------------------------------------------------------------
bool TTSWrapper::init(const std::string& model_path) {
    // Load JSON config (model_path.json)
    std::string config_path = model_path + ".json";
    if (!load_config(config_path)) {
        std::cerr << "TTS config not found: " << config_path << std::endl;
    }

#ifdef USE_ONNXRUNTIME
    try {
        env_ = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "TTS");

        std::ifstream mf(model_path);
        if (!mf.good()) {
            std::cerr << "TTS ONNX model not found: " << model_path << std::endl;
            return true;  // Will generate test tone
        }

        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);  // TTS is less compute-bound
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        session_ = new Ort::Session(*env_, model_path.c_str(), opts);
        initialized_ = true;
        std::cout << "TTS ONNX model loaded: " << model_path << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "TTS ONNX init error: " << e.what() << std::endl;
        return true;
    }
#endif

    return true;
}

// -----------------------------------------------------------------------
// Synthesize
// -----------------------------------------------------------------------
std::vector<float> TTSWrapper::synthesize(const std::string& hindi_text) {
    std::vector<float> audio;
    if (hindi_text.empty()) return audio;

#ifdef USE_ONNXRUNTIME
    if (initialized_ && session_) {
        try {
            // Convert Hindi text to phoneme IDs
            auto phoneme_ids = text_to_phoneme_ids(hindi_text);

            TTS_LOG("Phoneme IDs: %zu for text len %zu", phoneme_ids.size(), hindi_text.size());
            if (phoneme_ids.empty()) {
                TTS_LOG("No phoneme IDs generated!");
            } else {
                auto mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

                // input: phoneme IDs [1, phoneme_count]
                std::vector<int64_t> input_shape = {1, (int64_t)phoneme_ids.size()};
                Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
                    mem, phoneme_ids.data(), phoneme_ids.size(),
                    input_shape.data(), 2);

                // input_lengths: [1]
                std::vector<int64_t> lengths = {(int64_t)phoneme_ids.size()};
                std::vector<int64_t> len_shape = {1};
                Ort::Value len_tensor = Ort::Value::CreateTensor<int64_t>(
                    mem, lengths.data(), 1, len_shape.data(), 1);

                // scales: [3] = {noise_scale, length_scale, noise_w}
                std::vector<float> scales = {noise_scale_, length_scale_, noise_w_};
                std::vector<int64_t> scales_shape = {3};
                Ort::Value scales_tensor = Ort::Value::CreateTensor<float>(
                    mem, scales.data(), 3, scales_shape.data(), 1);

                // Run inference
                const char* input_names[] = {"input", "input_lengths", "scales"};
                const char* output_names[] = {"output"};

                std::vector<Ort::Value> inputs;
                inputs.push_back(std::move(input_tensor));
                inputs.push_back(std::move(len_tensor));
                inputs.push_back(std::move(scales_tensor));

                auto outputs = session_->Run(
                    Ort::RunOptions{nullptr},
                    input_names, inputs.data(), inputs.size(),
                    output_names, 1);

                // Extract audio [1, 1, samples]
                float* audio_data = outputs[0].GetTensorMutableData<float>();
                auto shape = outputs[0].GetTensorTypeAndShapeInfo().GetShape();
                size_t num_samples = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();

                std::vector<float> raw_audio(audio_data, audio_data + num_samples);

                // Resample from model rate to target rate
                audio = resample(raw_audio, model_sample_rate_, target_sample_rate_);
                return audio;
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "[TTS] ONNX inference error: " << e.what() << std::endl;
        }
    }
#endif

    // Fallback: generate test tone
    std::cerr << "[TTS] Generating test tone (no ONNX model)" << std::endl;
    size_t dur = target_sample_rate_ * 1;
    audio.resize(dur);
    for (size_t i = 0; i < dur; i++) {
        float t = static_cast<float>(i) / target_sample_rate_;
        audio[i] = 0.1f * std::sin(2.0f * M_PI * 440.0f * t) * std::exp(-t * 2.0f);
    }
    return audio;
}

std::vector<float> TTSWrapper::synthesize_chunk(const std::string& hindi_text,
                                                 size_t /*chunk_size_samples*/) {
    if (hindi_text.empty() && remaining_text_.empty()) return {};
    remaining_text_ += hindi_text;
    auto audio = synthesize(remaining_text_);
    remaining_text_.clear();
    return audio;
}
