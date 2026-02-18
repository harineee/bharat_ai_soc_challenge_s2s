/**
 * ASR wrapper for whisper.cpp streaming inference
 * Provides C++ interface for English speech recognition
 */

#ifndef ASR_WRAPPER_H
#define ASR_WRAPPER_H

#include <string>
#include <vector>
#include <memory>

// Forward declaration - actual whisper context hidden
struct whisper_context;

class ASRWrapper {
public:
    ASRWrapper();
    ~ASRWrapper();

    // Initialize with model path
    bool init(const std::string& model_path);

    // Process audio chunk (16 kHz, mono, float32)
    // Returns partial text result
    std::string process_chunk(const float* audio_samples, size_t num_samples);

    // Flush remaining audio buffer and get final result
    std::string flush();

    // Reset decoder state
    void reset();

    // Clear audio buffer (used after TTS playback to discard echo-contaminated audio)
    void clear_buffer();

    // Get current partial result
    std::string get_partial_text() const { return partial_text_; }

    // Configuration
    void set_chunk_size_ms(int ms) { chunk_size_ms_ = ms; }

private:
    whisper_context* ctx_;  // Raw pointer, freed in destructor
    std::string partial_text_;
    std::string confirmed_text_;  // Accumulated transcript across whisper runs
    int chunk_size_ms_;

    // Internal buffer for accumulating audio
    std::vector<float> audio_buffer_;
    static constexpr int SAMPLE_RATE = 16000;
};

#endif // ASR_WRAPPER_H
