/**
 * TTS wrapper — native ONNX Runtime inference for Piper VITS model.
 * NO Python subprocess calls.
 * Includes built-in Hindi (Devanagari) → IPA phonemization.
 */

#ifndef TTS_WRAPPER_H
#define TTS_WRAPPER_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

#ifdef USE_ONNXRUNTIME
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
#endif

class TTSWrapper {
public:
    TTSWrapper();
    ~TTSWrapper();

    bool init(const std::string& model_path);

    // Synthesize Hindi text → 16 kHz mono float32 PCM
    std::vector<float> synthesize(const std::string& hindi_text);

    // Chunked synthesis (returns full audio; streaming can be added later)
    std::vector<float> synthesize_chunk(const std::string& hindi_text,
                                        size_t chunk_size_samples);

    void set_sample_rate(int rate) { target_sample_rate_ = rate; }
    int get_sample_rate() const { return target_sample_rate_; }

private:
    // Piper model parameters (loaded from JSON config)
    int model_sample_rate_ = 22050;
    int target_sample_rate_ = 16000;
    float noise_scale_ = 0.667f;
    float length_scale_ = 1.0f;
    float noise_w_ = 0.8f;

    // Phoneme ID map: IPA char(s) → vector of int IDs
    std::unordered_map<std::string, std::vector<int64_t>> phoneme_id_map_;

    // Hindi Devanagari → IPA conversion table
    std::unordered_map<std::string, std::string> devanagari_to_ipa_;
    void init_devanagari_table();

    // Convert Hindi text to phoneme IDs for Piper
    std::vector<int64_t> text_to_phoneme_ids(const std::string& text);

    // Convert IPA string to phoneme IDs using the map
    std::vector<int64_t> ipa_to_ids(const std::string& ipa);

    // Load Piper JSON config
    bool load_config(const std::string& json_path);

    // Linear resample (model_sample_rate → target_sample_rate)
    std::vector<float> resample(const std::vector<float>& audio,
                                int from_rate, int to_rate);

#ifdef USE_ONNXRUNTIME
    Ort::Env* env_ = nullptr;
    Ort::Session* session_ = nullptr;
#endif

    bool initialized_ = false;
    std::string remaining_text_;
};

#endif // TTS_WRAPPER_H
