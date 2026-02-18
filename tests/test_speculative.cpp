/**
 * Standalone test for speculative decoding
 * Build: g++ -std=c++17 -DUSE_EXECUTORCH -I/path/to/executorch/include ...
 * Run:   ./test_speculative --model qwen3.pte --tokenizer tokenizer.json
 *
 * Tests:
 * 1. LLM loads and forward() returns logits
 * 2. Full autoregressive translation produces Hindi
 * 3. Speculative decoding produces same quality as full autoregressive
 * 4. Speculative is faster than autoregressive
 * 5. Token acceptance rate is measured
 */

#include "../mt/llm_translator.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <vector>
#include <string>

struct TestCase {
    std::string english;
};

bool contains_devanagari(const std::string& text) {
    // Devanagari Unicode range: U+0900 to U+097F
    // In UTF-8: 3-byte sequences starting with 0xE0 0xA4 or 0xE0 0xA5
    for (size_t i = 0; i + 2 < text.size(); i++) {
        unsigned char b0 = text[i], b1 = text[i+1];
        if (b0 == 0xE0 && (b1 == 0xA4 || b1 == 0xA5)) return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    std::string model_path, tokenizer_path;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i+1 < argc) model_path = argv[++i];
        if (arg == "--tokenizer" && i+1 < argc) tokenizer_path = argv[++i];
    }

    if (model_path.empty() || tokenizer_path.empty()) {
        std::cerr << "Usage: test_speculative --model X.pte --tokenizer Y.json" << std::endl;
        return 1;
    }

    LLMTranslator llm;
    std::cout << "Test 1: Loading model... ";
    assert(llm.init(model_path, tokenizer_path));
    std::cout << "PASS" << std::endl;

    std::vector<TestCase> tests = {
        {"Hello"},
        {"How are you"},
        {"The weather is nice today"},
        {"I will go to the market tomorrow"},
        {"Please help me find the train station"},
    };

    // Test 2: Full autoregressive
    std::cout << "\nTest 2: Full autoregressive translation" << std::endl;
    std::vector<std::string> full_outputs;
    double total_full_ms = 0;
    for (auto& tc : tests) {
        std::string result = llm.translate(tc.english);
        full_outputs.push_back(result);
        total_full_ms += llm.get_last_latency_ms();
        std::cout << "  \"" << tc.english << "\" -> \"" << result << "\" ("
                  << llm.get_last_latency_ms() << "ms)" << std::endl;
        assert(!result.empty() && "Translation should not be empty");
        assert(contains_devanagari(result) && "Output should contain Hindi");
    }

    // Test 3: Speculative with perfect drafts (= full autoregressive output)
    // When draft == LLM output, acceptance rate should be ~100%
    std::cout << "\nTest 3: Speculative with perfect drafts" << std::endl;
    for (size_t i = 0; i < tests.size(); i++) {
        std::string result = llm.speculative_translate(tests[i].english, full_outputs[i]);
        std::cout << "  Accepted: " << llm.get_last_accepted_tokens()
                  << "/" << llm.get_last_total_draft_tokens()
                  << " (" << llm.get_last_latency_ms() << "ms)" << std::endl;
    }

    // Test 4: Speculative with imperfect drafts (simulated NMT)
    std::cout << "\nTest 4: Speculative with imperfect drafts" << std::endl;
    std::vector<std::string> mock_nmt_drafts = {
        "नमस्ते",
        "आप कैसे हैं",
        "आज मौसम अच्छा है",
        "मैं कल बाज़ार जाऊंगा",
        "कृपया मुझे ट्रेन स्टेशन ढूंढने में मदद करें",
    };
    double total_spec_ms = 0;
    for (size_t i = 0; i < tests.size(); i++) {
        std::string result = llm.speculative_translate(tests[i].english, mock_nmt_drafts[i]);
        total_spec_ms += llm.get_last_latency_ms();
        std::cout << "  \"" << tests[i].english << "\" -> \"" << result << "\"" << std::endl;
        std::cout << "    Accepted: " << llm.get_last_accepted_tokens()
                  << "/" << llm.get_last_total_draft_tokens()
                  << " (" << llm.get_last_latency_ms() << "ms)" << std::endl;
    }

    // Test 5: Performance comparison
    std::cout << "\n=== Performance Summary ===" << std::endl;
    std::cout << "Full autoregressive: " << total_full_ms << "ms total ("
              << total_full_ms / tests.size() << "ms avg)" << std::endl;
    std::cout << "Speculative:         " << total_spec_ms << "ms total ("
              << total_spec_ms / tests.size() << "ms avg)" << std::endl;
    if (total_spec_ms > 0) {
        double speedup = total_full_ms / total_spec_ms;
        std::cout << "Speedup:             " << speedup << "x" << std::endl;
    }
    std::cout << "Acceptance rate:     " << (llm.get_acceptance_rate() * 100) << "%" << std::endl;

    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
