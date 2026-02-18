#include "asr_wrapper.h"
#include "whisper.h" // whisper.cpp header

#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

#ifdef __ANDROID__
#include <android/log.h>
#define ASR_LOG(...) __android_log_print(ANDROID_LOG_INFO, "ASR", __VA_ARGS__)
#else
#define ASR_LOG(...) do { fprintf(stderr, "[ASR] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif

ASRWrapper::ASRWrapper() : chunk_size_ms_(80) {
    ctx_ = nullptr;
}

ASRWrapper::~ASRWrapper() {
    if (ctx_) {
        whisper_free(ctx_);
    }
}

bool ASRWrapper::init(const std::string& model_path) {
    // Load whisper model
    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // CPU only
    
    whisper_context* ctx = whisper_init_from_file_with_params(
        model_path.c_str(), cparams);
    
    if (!ctx) {
        return false;
    }
    
    ctx_ = ctx;
    return true;
}

std::string ASRWrapper::process_chunk(const float* audio_samples, 
                                      size_t num_samples) {
    if (!ctx_) {
        return "";
    }
    
    // Accumulate samples
    size_t old_size = audio_buffer_.size();
    audio_buffer_.resize(old_size + num_samples);
    std::memcpy(audio_buffer_.data() + old_size, 
                audio_samples, 
                num_samples * sizeof(float));
    
    // Process when we have enough samples for a chunk
    // Use 3 seconds — fewer whisper invocations = better throughput (whisper has ~2s fixed overhead)
    size_t min_samples = SAMPLE_RATE * 3; // 3 seconds
    size_t chunk_samples = std::max(static_cast<size_t>((chunk_size_ms_ * SAMPLE_RATE) / 1000), min_samples);

    // Cap buffer to 10 seconds to keep processing time reasonable on mobile
    static const size_t kMaxSamples = SAMPLE_RATE * 10;
    if (audio_buffer_.size() > kMaxSamples) {
        audio_buffer_.erase(audio_buffer_.begin(),
                            audio_buffer_.begin() + (audio_buffer_.size() - kMaxSamples));
    }

    if (audio_buffer_.size() >= chunk_samples) {
        // Simple amplitude-based VAD: skip whisper entirely on silence
        float peak = 0.0f;
        for (size_t i = 0; i < audio_buffer_.size(); i++) {
            float v = std::abs(audio_buffer_[i]);
            if (v > peak) peak = v;
        }
        if (peak < 0.01f) {
            ASR_LOG("VAD: silence detected (peak=%.4f), skipping whisper", peak);
            audio_buffer_.clear();
            return partial_text_;
        }

        ASR_LOG("Running whisper: buffer=%zu samples (%.1fs), peak=%.3f",
                audio_buffer_.size(), (float)audio_buffer_.size() / SAMPLE_RATE, peak);
        whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
        params.print_progress = false;
        params.print_special = false;
        params.print_realtime = false;
        params.translate = false; // English ASR, not translation
        params.language = "en";
        params.n_threads = 8; // 8 threads for 8-core ARM — base.en needs more compute
        params.offset_ms = 0;
        params.no_context = true;  // Don't use cross-chunk context (faster)
        params.single_segment = true; // Single segment output (faster)
        params.suppress_blank = true;
        params.temperature = 0.0f; // Deterministic
        params.audio_ctx = 512;    // Reduce from 1500 (30s) — 512 (~10s) is enough for 3s chunks
        params.n_max_text_ctx = 0; // No text context needed
        params.max_tokens = 32;    // Cap decoding length — prevents runaway on noisy input

        size_t n = audio_buffer_.size();
        if (n == 0) return partial_text_;
        // Run inference
        int result = whisper_full(ctx_, params, 
                                 audio_buffer_.data(), 
                                 static_cast<int>(n));
        
        ASR_LOG("whisper_full returned %d", result);
        if (result == 0) {
            int n_segments = whisper_full_n_segments(ctx_);
            std::string text;
            
            for (int i = 0; i < n_segments; i++) {
                const char* segment_text = whisper_full_get_segment_text(ctx_, i);
                if (segment_text) {
                    std::string seg_str(segment_text);
                    // Trim whitespace
                    while (!seg_str.empty() && (std::isspace(seg_str[0]) || seg_str[0] == '\0')) {
                        seg_str.erase(0, 1);
                    }
                    while (!seg_str.empty() && (std::isspace(seg_str.back()) || seg_str.back() == '\0')) {
                        seg_str.pop_back();
                    }
                    if (!seg_str.empty() && seg_str != "[BLANK_AUDIO]") {
                        if (!text.empty()) text += " ";
                        text += seg_str;
                    }
                }
            }
            
            ASR_LOG("whisper result: %d segments, text=\"%s\"", n_segments, text.c_str());

            // Filter common whisper-tiny hallucinations on noise/silence
            if (!text.empty()) {
                bool is_hallucination = false;
                // Check for bracket/paren patterns like [MUSIC], (speaking in...)
                if (text.front() == '[' || text.front() == '(') {
                    is_hallucination = true;
                }
                // Common single-word hallucinations
                std::string lower_text = text;
                std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
                static const char* hallucinations[] = {
                    "you", "thank you", "thank you.", "thanks for watching",
                    "thanks for watching.", "bye", "bye.", "goodbye",
                    "the end", "the end.", "so", "okay",
                    nullptr
                };
                for (const char** h = hallucinations; *h; ++h) {
                    if (lower_text == *h) {
                        is_hallucination = true;
                        break;
                    }
                }
                if (is_hallucination) {
                    ASR_LOG("Filtered hallucination: \"%s\"", text.c_str());
                    text.clear();
                }
            }

            if (!text.empty()) {
                // Accumulate text across whisper runs (keep full context)
                if (confirmed_text_.empty()) {
                    confirmed_text_ = text;
                } else {
                    confirmed_text_ += " " + text;
                }
                partial_text_ = confirmed_text_;
                ASR_LOG("Accumulated transcript: \"%s\"", confirmed_text_.c_str());
            }

            // Clear entire buffer after processing (no overlap needed
            // since no_context=true — each run is independent)
            audio_buffer_.clear();
        }
    }
    
    return partial_text_;
}

std::string ASRWrapper::flush() {
    // Force process remaining audio buffer even if it's smaller than chunk size
    if (!ctx_ || audio_buffer_.empty()) {
        return partial_text_;
    }
    size_t n = audio_buffer_.size();
    static const size_t kMaxSamples = SAMPLE_RATE * 10;
    if (n > kMaxSamples) n = kMaxSamples;

    ASR_LOG("Flushing whisper: %zu samples (%.1fs)", n, (float)n / SAMPLE_RATE);

    // Process remaining audio (even if less than minimum)
    whisper_full_params params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    params.print_progress = false;
    params.print_special = false;
    params.print_realtime = false;
    params.translate = false;
    params.language = "en";
    params.n_threads = 6;
    params.offset_ms = 0;
    params.no_context = true;
    params.single_segment = true;
    params.suppress_blank = true;
    params.temperature = 0.0f;
    params.audio_ctx = 512;
    params.n_max_text_ctx = 0;
    params.max_tokens = 32;
    
    int result = whisper_full(ctx_, params, 
                             audio_buffer_.data(), 
                             static_cast<int>(n));
    
    if (result == 0) {
        int n_segments = whisper_full_n_segments(ctx_);
        std::string text;
        
        for (int i = 0; i < n_segments; i++) {
            const char* segment_text = whisper_full_get_segment_text(ctx_, i);
            if (segment_text) {
                std::string seg_str(segment_text);
                while (!seg_str.empty() && (std::isspace(seg_str[0]) || seg_str[0] == '\0')) {
                    seg_str.erase(0, 1);
                }
                while (!seg_str.empty() && (std::isspace(seg_str.back()) || seg_str.back() == '\0')) {
                    seg_str.pop_back();
                }
                if (!seg_str.empty() && seg_str != "[BLANK_AUDIO]") {
                    if (!text.empty()) text += " ";
                    text += seg_str;
                }
            }
        }
        
        // Filter hallucinations (same as process_chunk)
        if (!text.empty()) {
            bool is_hallucination = false;
            if (text.front() == '[' || text.front() == '(') {
                is_hallucination = true;
            }
            std::string lower_text = text;
            std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
            static const char* hallucinations[] = {
                "you", "thank you", "thank you.", "thanks for watching",
                "thanks for watching.", "bye", "bye.", "goodbye",
                "the end", "the end.", "so", "okay",
                nullptr
            };
            for (const char** h = hallucinations; *h; ++h) {
                if (lower_text == *h) {
                    is_hallucination = true;
                    break;
                }
            }
            if (is_hallucination) {
                ASR_LOG("Filtered hallucination in flush: \"%s\"", text.c_str());
                text.clear();
            }
        }

        if (!text.empty()) {
            if (confirmed_text_.empty()) {
                confirmed_text_ = text;
            } else {
                confirmed_text_ += " " + text;
            }
            partial_text_ = confirmed_text_;
        }
    }

    audio_buffer_.clear();
    return partial_text_;
}

void ASRWrapper::reset() {
    if (ctx_) {
        whisper_reset_timings(ctx_);
    }
    partial_text_.clear();
    confirmed_text_.clear();
    audio_buffer_.clear();
}

void ASRWrapper::clear_buffer() {
    audio_buffer_.clear();
    ASR_LOG("Audio buffer cleared (echo suppression flush)");
}
