/**
 * Phrase boundary detection for triggering MT translation
 * Detects natural pause points in speech
 */

#ifndef PHRASE_DETECTOR_H
#define PHRASE_DETECTOR_H

#include <string>
#include <chrono>
#include <vector>

struct PhraseBoundary {
    std::string text;
    bool is_complete;
    std::chrono::milliseconds timestamp;
};

class PhraseDetector {
public:
    PhraseDetector()
        : pause_threshold_ms_(150)
        , min_words_(1)
        , max_words_(8)
        , last_update_time_(std::chrono::steady_clock::now())
    {}

    // Process incremental ASR text
    // Returns true if a phrase boundary is detected
    bool process_text(const std::string& partial_text, 
                     std::vector<PhraseBoundary>& boundaries);

    // Reset detector state
    void reset();

    // Configuration
    void set_pause_threshold_ms(int ms) { pause_threshold_ms_ = ms; }
    void set_min_words(int words) { min_words_ = words; }
    void set_max_words(int words) { max_words_ = words; }

private:
    int pause_threshold_ms_;
    int min_words_;
    int max_words_;
    
    std::string accumulated_text_;
    std::string last_sent_text_;  // Track last text sent to MT to avoid duplicates
    std::chrono::steady_clock::time_point last_update_time_;

    // Helper functions
    int count_words(const std::string& text);
    bool has_punctuation(const std::string& text);
    std::string extract_phrase();
};

#endif // PHRASE_DETECTOR_H
