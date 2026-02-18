/**
 * Translation mode enum — controls quality/speed tradeoff
 */
#ifndef TRANSLATION_MODE_H
#define TRANSLATION_MODE_H

enum class TranslationMode {
    SPEED,      // Marian NMT only (~49ms, correct Hindi)
    BALANCED,   // NMT draft + LLM speculative verify (65% acceptance)
    QUALITY     // NMT draft + LLM refinement (same path as Balanced)
};

#endif // TRANSLATION_MODE_H
