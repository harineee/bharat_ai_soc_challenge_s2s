#include "phrase_detector.h"
#include <sstream>
#include <cctype>
#include <algorithm>

bool PhraseDetector::process_text(const std::string& partial_text,
                                  std::vector<PhraseBoundary>& boundaries) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - last_update_time_);
    
    bool boundary_detected = false;
    
    // Check for pause (no update for threshold duration)
    if (elapsed.count() > pause_threshold_ms_ && !accumulated_text_.empty()) {
        int word_count = count_words(accumulated_text_);

        // Trigger if we have enough words or punctuation,
        // AND the text is different from what we last sent to MT
        if ((word_count >= min_words_ || has_punctuation(accumulated_text_))
            && accumulated_text_ != last_sent_text_) {
            PhraseBoundary boundary;
            boundary.text = extract_phrase();
            boundary.is_complete = has_punctuation(accumulated_text_);
            boundary.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch());

            boundaries.push_back(boundary);
            last_sent_text_ = accumulated_text_;
            accumulated_text_.clear();
            boundary_detected = true;
        }
    }

    // Update accumulated text
    if (partial_text != accumulated_text_) {
        accumulated_text_ = partial_text;
        last_update_time_ = now;
    }

    // Check for max word count (force boundary)
    int word_count = count_words(accumulated_text_);
    if (word_count >= max_words_ && !boundary_detected
        && accumulated_text_ != last_sent_text_) {
        PhraseBoundary boundary;
        boundary.text = extract_phrase();
        boundary.is_complete = false; // Forced boundary
        boundary.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch());

        boundaries.push_back(boundary);
        last_sent_text_ = accumulated_text_;
        accumulated_text_.clear();
        boundary_detected = true;
    }
    
    return boundary_detected;
}

void PhraseDetector::reset() {
    accumulated_text_.clear();
    last_sent_text_.clear();
    last_update_time_ = std::chrono::steady_clock::now();
}

int PhraseDetector::count_words(const std::string& text) {
    if (text.empty()) return 0;
    
    std::istringstream iss(text);
    int count = 0;
    std::string word;
    while (iss >> word) {
        count++;
    }
    return count;
}

bool PhraseDetector::has_punctuation(const std::string& text) {
    const std::string punctuation = ".,!?;:";
    return std::any_of(text.begin(), text.end(), [&punctuation](char c) {
        return punctuation.find(c) != std::string::npos;
    });
}

std::string PhraseDetector::extract_phrase() {
    // Extract phrase up to punctuation or word limit
    std::istringstream iss(accumulated_text_);
    std::ostringstream oss;
    int word_count = 0;
    std::string word;
    
    while (iss >> word && word_count < max_words_) {
        if (word_count > 0) oss << " ";
        oss << word;
        word_count++;
        
        // Stop at punctuation
        if (has_punctuation(word)) {
            break;
        }
    }
    
    return oss.str();
}
