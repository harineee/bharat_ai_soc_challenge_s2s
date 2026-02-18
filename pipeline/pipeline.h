/**
 * Main pipeline orchestrator
 * Manages parallel execution of ASR → MT → TTS → Playback
 */

#ifndef PIPELINE_H
#define PIPELINE_H

#include "asr_wrapper.h"
#include "mt_wrapper.h"
#include "tts_wrapper.h"
#include "phrase_detector.h"
#include "lockfree_queue.h"
#include "performance_tracker.h"

#include <thread>
#include <atomic>
#include <mutex>
#include <memory>
#include <vector>
#include <string>

struct PipelineConfig {
    std::string asr_model_path;
    std::string mt_model_path;
    std::string tts_model_path;

    // LLM translation (ExecuTorch + Qwen3-0.6B)
    std::string llm_model_path;       // Path to .pte file (empty = disabled)
    std::string llm_tokenizer_path;   // Path to tokenizer.json

    // Translation mode: 0=SPEED (NMT), 1=BALANCED (speculative), 2=QUALITY (refinement)
    int translation_mode = 0;  // Default to Speed for responsive demo

    int sample_rate = 16000;
    int chunk_size_ms = 80;

    // Threading
    bool pin_threads = true;
    int asr_thread_priority = 10;
    int mt_thread_priority = 5;
    int tts_thread_priority = 5;
};

class Pipeline {
public:
    Pipeline();
    ~Pipeline();

    // Initialize pipeline with models
    bool init(const PipelineConfig& config);

    // Start pipeline (spawns threads)
    bool start();

    // Stop pipeline (joins threads)
    void stop();

    // Push audio samples (called from audio capture thread)
    void push_audio(const float* samples, size_t num_samples);

    // Get synthesized audio (called from playback thread)
    bool pop_audio(std::vector<float>& audio_samples);

    // Flush ASR to process remaining audio
    void flush_asr();

    // Wait for audio queue to drain (all audio processed by ASR)
    void wait_audio_drained(int timeout_ms = 5000);

    // Get current status
    bool is_running() const { return running_; }
    std::string get_current_english() const {
        std::lock_guard<std::mutex> lock(text_mutex_);
        return current_english_;
    }
    std::string get_current_hindi() const {
        std::lock_guard<std::mutex> lock(text_mutex_);
        return current_hindi_;
    }
    
    // Check if LLM backend is active
    bool is_llm_active() const;

    // Translation mode control
    void set_translation_mode(int mode);
    std::string get_translation_mode_name() const;
    double get_mt_acceptance_rate() const;

    // Performance tracking
    PerformanceTracker* get_performance_tracker() { return perf_tracker_.get(); }
    void enable_performance_tracking(bool enable);

private:
    // Thread functions
    void asr_thread_func();
    void mt_thread_func();
    void tts_thread_func();

    // Components
    std::unique_ptr<ASRWrapper> asr_;
    std::unique_ptr<MTWrapper> mt_;
    std::unique_ptr<TTSWrapper> tts_;
    std::unique_ptr<PhraseDetector> phrase_detector_;
    std::unique_ptr<PerformanceTracker> perf_tracker_;

    // Queues (SPSC - Single Producer Single Consumer)
    LockFreeQueue<std::vector<float>> audio_queue_;      // Audio → ASR
    LockFreeQueue<std::string> asr_text_queue_;          // ASR → MT
    LockFreeQueue<std::string> mt_text_queue_;           // MT → TTS
    LockFreeQueue<std::vector<float>> tts_audio_queue_;  // TTS → Playback

    // Threads
    std::thread asr_thread_;
    std::thread mt_thread_;
    std::thread tts_thread_;

    // State
    std::atomic<bool> running_;
    mutable std::mutex text_mutex_;
    std::string current_english_;
    std::string current_hindi_;

    // Sentence-wise translation tracking
    std::atomic<size_t> last_translated_pos_{0};  // Position up to which ASR text has been sent to MT

    // Echo suppression: mute mic while TTS is playing through speaker
    std::atomic<bool> suppress_audio_{false};
    std::mutex suppress_mutex_;
    std::chrono::steady_clock::time_point suppress_until_;

    PipelineConfig config_;
};

#endif // PIPELINE_H
