#include "pipeline.h"
#include "translation_mode.h"
#include "phrase_detector.h"
#include "performance_tracker.h"
#include <iostream>
#include <chrono>
#include <thread>

#ifdef __ANDROID__
#include <sys/syscall.h>
#include <unistd.h>
#include <android/log.h>
// Write to both logcat and a file (OnePlus suppresses logcat)
static FILE* g_pipe_log = nullptr;
static void pipe_log_impl(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    __android_log_vprint(ANDROID_LOG_INFO, "Pipeline", fmt, args);
    va_end(args);
    if (g_pipe_log) {
        va_list args2;
        va_start(args2, fmt);
        vfprintf(g_pipe_log, fmt, args2);
        va_end(args2);
        fprintf(g_pipe_log, "\n");
        fflush(g_pipe_log);
    }
}
#define PIPE_LOG(...) pipe_log_impl(__VA_ARGS__)
#else
#define PIPE_LOG(...) do { fprintf(stderr, "[Pipeline] "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } while(0)
#endif
#include <cstdarg>

Pipeline::Pipeline()
    : audio_queue_(1024)      // Buffer ~1 second at 16 kHz
    , asr_text_queue_(64)     // Text phrases
    , mt_text_queue_(64)      // Translated phrases
    , tts_audio_queue_(32)    // Audio chunks
    , running_(false)
{
}

Pipeline::~Pipeline() {
    stop();
}

bool Pipeline::init(const PipelineConfig& config) {
    config_ = config;

#ifdef __ANDROID__
    // Open file log (OnePlus devices suppress logcat)
    {
        std::string log_path = config.asr_model_path;
        size_t pos = log_path.find("/files/");
        if (pos != std::string::npos) {
            std::string lf = log_path.substr(0, pos) + "/files/pipeline.log";
            g_pipe_log = fopen(lf.c_str(), "w");
        }
    }
#endif
    PIPE_LOG("Pipeline::init() started");

    // Initialize performance tracker
    perf_tracker_ = std::make_unique<PerformanceTracker>();
    if (!g_performance_tracker) {
        g_performance_tracker = perf_tracker_.get();
    }
    
    // Initialize ASR
    {
        PERF_START("asr_init");
        PIPE_LOG("Initializing ASR: %s", config.asr_model_path.c_str());
        asr_ = std::make_unique<ASRWrapper>();
        if (!asr_->init(config.asr_model_path)) {
            PIPE_LOG("ASR init FAILED");
            return false;
        }
        asr_->set_chunk_size_ms(config.chunk_size_ms);
        PIPE_LOG("ASR init OK");
    }

    // Initialize MT (Marian NMT)
    {
        PERF_START("mt_init");
        PIPE_LOG("Initializing MT: %s", config.mt_model_path.c_str());
        mt_ = std::make_unique<MTWrapper>();
        if (!mt_->init(config.mt_model_path)) {
            PIPE_LOG("MT init FAILED (will use placeholder)");
        } else {
            PIPE_LOG("MT init OK (nmt_active=%d)", mt_->is_nmt_active() ? 1 : 0);
        }
    }

    // Initialize LLM translator (if path provided)
    if (!config.llm_model_path.empty()) {
        PERF_START("llm_init");
        PIPE_LOG("Initializing LLM: %s (tokenizer: %s)",
                 config.llm_model_path.c_str(), config.llm_tokenizer_path.c_str());
        if (mt_->init_llm(config.llm_model_path, config.llm_tokenizer_path)) {
            PIPE_LOG("LLM init OK (llm_active=%d)", mt_->is_llm_active() ? 1 : 0);
        } else {
            PIPE_LOG("LLM init FAILED — using NMT fallback");
        }
    } else {
        PIPE_LOG("LLM model path empty — skipping LLM init");
    }

    // Set translation mode from config
    set_translation_mode(config.translation_mode);

    // Initialize TTS
    {
        PERF_START("tts_init");
        tts_ = std::make_unique<TTSWrapper>();
        if (!tts_->init(config.tts_model_path)) {
            std::cerr << "Failed to initialize TTS" << std::endl;
            // Don't fail, will use placeholder
        }
        tts_->set_sample_rate(config.sample_rate);
    }
    
    // Initialize phrase detector
    phrase_detector_ = std::make_unique<PhraseDetector>();
    
    return true;
}

bool Pipeline::start() {
    if (running_.load()) {
        return false;
    }
    
    running_.store(true);
    last_translated_pos_.store(0);

    // Reset accumulated text for new session
    {
        std::lock_guard<std::mutex> lock(text_mutex_);
        current_english_.clear();
        current_hindi_.clear();
    }
    if (asr_) asr_->reset();

    // Start ASR thread
    asr_thread_ = std::thread([this]() { this->asr_thread_func(); });
    
    // Start MT thread
    mt_thread_ = std::thread([this]() { this->mt_thread_func(); });
    
    // Start TTS thread
    tts_thread_ = std::thread([this]() { this->tts_thread_func(); });
    
    // Set thread priorities (Android-specific)
#ifdef __ANDROID__
    if (config_.pin_threads) {
        // Note: Requires root or appropriate permissions
        // For production, use Android's native thread priority APIs
    }
#endif
    
    return true;
}

void Pipeline::stop() {
    if (!running_.load()) {
        return;
    }

    PIPE_LOG("Stopping pipeline — flushing ASR first...");

    // Stop accepting new audio but let threads finish current work
    running_.store(false);

    // Wait for ASR thread to finish (may be inside whisper_full)
    if (asr_thread_.joinable()) {
        asr_thread_.join();
    }

    // After ASR thread exits, flush any UNTRANSLATED remainder to MT
    {
        std::string cur_eng;
        {
            std::lock_guard<std::mutex> lock(text_mutex_);
            cur_eng = current_english_;
        }
        size_t pos = last_translated_pos_.load();
        if (pos < cur_eng.size()) {
            std::string remainder = cur_eng.substr(pos);
            // Trim whitespace
            size_t start = remainder.find_first_not_of(" \t\n");
            if (start != std::string::npos && start > 0) {
                remainder = remainder.substr(start);
            }
            if (!remainder.empty() && remainder != "[BLANK_AUDIO]") {
                PIPE_LOG("Flushing remainder → MT: \"%s\"", remainder.c_str());
                asr_text_queue_.push(remainder);
            }
        }
    }

    // Process any remaining items in the MT queue
    std::string final_english;
    while (asr_text_queue_.pop(final_english)) {
        if (final_english.empty()) continue;
        PIPE_LOG("Final MT translate: \"%s\"", final_english.c_str());
        std::string hindi = mt_->translate(final_english);
        if (!hindi.empty()) {
            PIPE_LOG("Final MT result: \"%s\"", hindi.c_str());
            {
                std::lock_guard<std::mutex> lock(text_mutex_);
                current_hindi_ += " " + hindi;
            }
            mt_text_queue_.push(hindi);
        }
    }

    // Now stop MT and TTS threads
    if (mt_thread_.joinable()) {
        mt_thread_.join();
    }
    if (tts_thread_.joinable()) {
        tts_thread_.join();
    }

    PIPE_LOG("Pipeline stopped");

    // Print performance summary
    if (perf_tracker_) {
        perf_tracker_->print_summary();
    }
}

void Pipeline::enable_performance_tracking(bool enable) {
    if (enable && !perf_tracker_) {
        perf_tracker_ = std::make_unique<PerformanceTracker>();
        g_performance_tracker = perf_tracker_.get();
    } else if (!enable) {
        perf_tracker_.reset();
        if (g_performance_tracker == perf_tracker_.get()) {
            g_performance_tracker = nullptr;
        }
    }
}

void Pipeline::push_audio(const float* samples, size_t num_samples) {
    // Echo suppression: drop mic audio while TTS is playing through speaker
    if (suppress_audio_.load()) {
        bool still_suppressed;
        {
            std::lock_guard<std::mutex> lock(suppress_mutex_);
            still_suppressed = std::chrono::steady_clock::now() < suppress_until_;
        }
        if (still_suppressed) {
            return;  // Drop audio — speaker is playing TTS
        }
        suppress_audio_.store(false);
        // Flush ASR buffer — any audio accumulated during/after TTS playback
        // is likely contaminated with speaker echo
        if (asr_) {
            asr_->clear_buffer();
        }
    }

    std::vector<float> audio_chunk(samples, samples + num_samples);
    audio_queue_.push(audio_chunk);
}

bool Pipeline::pop_audio(std::vector<float>& audio_samples) {
    return tts_audio_queue_.pop(audio_samples);
}

void Pipeline::asr_thread_func() {
    std::vector<float> audio_chunk;
    PIPE_LOG("ASR thread started");
    int chunk_count = 0;

    while (running_.load()) {
        if (audio_queue_.pop(audio_chunk)) {
            PERF_START("asr_process");
            chunk_count++;

            // Process audio with ASR (returns accumulated transcript)
            auto asr_t0 = std::chrono::steady_clock::now();
            std::string full_text = asr_->process_chunk(
                audio_chunk.data(),
                audio_chunk.size());
            auto asr_t1 = std::chrono::steady_clock::now();
            auto asr_ms = std::chrono::duration_cast<std::chrono::milliseconds>(asr_t1 - asr_t0).count();
            float audio_sec = (float)audio_chunk.size() / 16000.0f;
            PIPE_LOG("ASR chunk: %zu samples (%.1fs audio) → %ldms (%.2fx RT)",
                     audio_chunk.size(), audio_sec, asr_ms,
                     asr_ms > 0 ? (audio_sec * 1000.0f / asr_ms) : 0.0f);

            if (!full_text.empty() && full_text != "[BLANK_AUDIO]") {
                {
                    std::lock_guard<std::mutex> lock(text_mutex_);
                    current_english_ = full_text;
                }

                // Sentence-wise detection: look for . ? ! after last_translated_pos_
                size_t pos = last_translated_pos_.load();
                while (pos < full_text.size()) {
                    // Find next sentence-ending punctuation
                    size_t sent_end = std::string::npos;
                    for (size_t j = pos; j < full_text.size(); j++) {
                        char c = full_text[j];
                        if (c == '.' || c == '?' || c == '!') {
                            sent_end = j;
                            break;
                        }
                    }

                    if (sent_end == std::string::npos) break;  // No complete sentence yet

                    // Extract sentence
                    std::string sentence = full_text.substr(pos, sent_end - pos + 1);

                    // Trim leading whitespace
                    size_t start = sentence.find_first_not_of(" \t\n");
                    if (start != std::string::npos && start > 0) {
                        sentence = sentence.substr(start);
                    }

                    // Send to MT if non-empty and not noise
                    if (!sentence.empty() && sentence.size() > 1 && sentence != "[BLANK_AUDIO]") {
                        PIPE_LOG("Sentence → MT: \"%s\"", sentence.c_str());
                        asr_text_queue_.push(sentence);
                    }

                    // Advance past this sentence + trailing whitespace
                    pos = sent_end + 1;
                    while (pos < full_text.size() && full_text[pos] == ' ') pos++;
                    last_translated_pos_.store(pos);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    PIPE_LOG("ASR thread stopped (%d chunks processed)", chunk_count);
}

void Pipeline::mt_thread_func() {
    std::string english_text;
    PIPE_LOG("MT thread started (mode=%s)", get_translation_mode_name().c_str());

    while (running_.load()) {
        if (asr_text_queue_.pop(english_text)) {
            PERF_START("mt_translate");

            PIPE_LOG("MT translating: \"%s\"", english_text.c_str());
            std::string hindi_text = mt_->translate(english_text);

            if (!hindi_text.empty()) {
                PIPE_LOG("MT result: \"%s\"", hindi_text.c_str());
                {
                    std::lock_guard<std::mutex> lock(text_mutex_);
                    if (current_hindi_.empty()) {
                        current_hindi_ = hindi_text;
                    } else {
                        current_hindi_ += " " + hindi_text;
                    }
                }
                mt_text_queue_.push(hindi_text);
            } else {
                PIPE_LOG("MT returned empty for: \"%s\"", english_text.c_str());
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    PIPE_LOG("MT thread stopped");
}

void Pipeline::tts_thread_func() {
    std::string hindi_text;
    size_t chunk_samples = (config_.sample_rate * 500) / 1000; // 500 ms chunks
    PIPE_LOG("TTS thread started");

    while (running_.load()) {
        if (mt_text_queue_.pop(hindi_text)) {
            PERF_START("tts_synthesize");

            PIPE_LOG("TTS synthesizing: \"%s\"", hindi_text.c_str());
            std::vector<float> audio_chunk = tts_->synthesize_chunk(
                hindi_text,
                chunk_samples);

            if (!audio_chunk.empty()) {
                PIPE_LOG("TTS output: %zu samples", audio_chunk.size());
                tts_audio_queue_.push(audio_chunk);

                // Suppress mic input for duration of TTS playback to prevent echo
                double duration_sec = (double)audio_chunk.size() / config_.sample_rate;
                int suppress_ms = (int)(duration_sec * 1000) + 1000;  // +1000ms buffer for AudioTrack drain + room reverb
                {
                    std::lock_guard<std::mutex> lock(suppress_mutex_);
                    auto new_until = std::chrono::steady_clock::now()
                        + std::chrono::milliseconds(suppress_ms);
                    if (new_until > suppress_until_) {
                        suppress_until_ = new_until;
                    }
                }
                suppress_audio_.store(true);
                PIPE_LOG("Mic suppressed for %dms (TTS playback)", suppress_ms);
            } else {
                PIPE_LOG("TTS returned empty audio");
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    PIPE_LOG("TTS thread stopped");
}

bool Pipeline::is_llm_active() const {
    return mt_ && mt_->is_llm_active();
}

void Pipeline::set_translation_mode(int mode) {
    if (!mt_) return;
    switch (mode) {
        case 0: mt_->set_translation_mode(TranslationMode::SPEED); break;
        case 1: mt_->set_translation_mode(TranslationMode::BALANCED); break;
        case 2: mt_->set_translation_mode(TranslationMode::QUALITY); break;
        default:
            std::cerr << "Unknown translation mode " << mode << ", using BALANCED" << std::endl;
            mt_->set_translation_mode(TranslationMode::BALANCED);
            break;
    }
}

std::string Pipeline::get_translation_mode_name() const {
    if (!mt_) return "Unknown";
    return mt_->get_mode_name();
}

double Pipeline::get_mt_acceptance_rate() const {
    if (!mt_) return 0.0;
    return mt_->get_acceptance_rate();
}

void Pipeline::wait_audio_drained(int timeout_ms) {
    int waited = 0;
    while (!audio_queue_.empty() && waited < timeout_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        waited += 10;
    }
    // Extra wait for ASR to finish processing the last chunk
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

void Pipeline::flush_asr() {
    // Flush ASR to process remaining audio
    if (asr_) {
        std::string final_text = asr_->flush();

        std::string cur_eng;
        {
            std::lock_guard<std::mutex> lock(text_mutex_);
            cur_eng = current_english_;
        }

        // Use the best available text: flush result if non-empty, else current_english_
        if (final_text.empty() || final_text == "[BLANK_AUDIO]") {
            final_text = cur_eng;
        }

        // If we have accumulated English text that hasn't been translated yet,
        // push the full text (not just the tail from the remaining buffer)
        if (!cur_eng.empty() && cur_eng != "[BLANK_AUDIO]") {
            asr_text_queue_.push(cur_eng);
        } else if (!final_text.empty() && final_text != "[BLANK_AUDIO]") {
            {
                std::lock_guard<std::mutex> lock(text_mutex_);
                current_english_ = final_text;
            }
            asr_text_queue_.push(final_text);
        }
    }
}
