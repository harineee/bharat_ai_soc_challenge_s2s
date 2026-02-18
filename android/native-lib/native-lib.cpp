/**
 * JNI bridge for Android
 * Provides Java interface to C++ pipeline
 */

#include <jni.h>
#include <string>
#include <memory>
#include <android/log.h>
#include "pipeline/pipeline.h"

#define LOG_TAG "NativePipeline"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

// File-based logging for OnePlus devices that suppress logcat
static FILE* g_logfile = nullptr;
static void file_log(const char* fmt, ...) {
    if (!g_logfile) return;
    va_list args;
    va_start(args, fmt);
    vfprintf(g_logfile, fmt, args);
    va_end(args);
    fprintf(g_logfile, "\n");
    fflush(g_logfile);
}
#include <cstdarg>

static std::unique_ptr<Pipeline> g_pipeline = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_armm_translation_NativePipeline_nativeInit(
    JNIEnv *env,
    jobject /* this */,
    jstring asr_model_path,
    jstring mt_model_path,
    jstring tts_model_path,
    jstring llm_model_path,
    jstring llm_tokenizer_path,
    jint translation_mode) {

    if (g_pipeline) {
        LOGE("Pipeline already initialized");
        return JNI_FALSE; // Already initialized
    }

    LOGI("Initializing pipeline...");
    g_pipeline = std::make_unique<Pipeline>();

    PipelineConfig config;

    // Convert Java strings to C++ strings
    const char* asr_path = env->GetStringUTFChars(asr_model_path, nullptr);
    const char* mt_path = env->GetStringUTFChars(mt_model_path, nullptr);
    const char* tts_path = env->GetStringUTFChars(tts_model_path, nullptr);
    const char* llm_path = llm_model_path ? env->GetStringUTFChars(llm_model_path, nullptr) : nullptr;
    const char* llm_tok_path = llm_tokenizer_path ? env->GetStringUTFChars(llm_tokenizer_path, nullptr) : nullptr;

    config.asr_model_path = asr_path ? asr_path : "";
    config.mt_model_path = mt_path ? mt_path : "";
    config.tts_model_path = tts_path ? tts_path : "";
    config.llm_model_path = llm_path ? llm_path : "";
    config.llm_tokenizer_path = llm_tok_path ? llm_tok_path : "";
    config.translation_mode = static_cast<int>(translation_mode);

    LOGI("Model paths - ASR: %s, MT: %s, TTS: %s",
         config.asr_model_path.c_str(),
         config.mt_model_path.c_str(),
         config.tts_model_path.c_str());
    LOGI("LLM model: %s", config.llm_model_path.c_str());

    env->ReleaseStringUTFChars(asr_model_path, asr_path);
    env->ReleaseStringUTFChars(mt_model_path, mt_path);
    env->ReleaseStringUTFChars(tts_model_path, tts_path);
    if (llm_path) env->ReleaseStringUTFChars(llm_model_path, llm_path);
    if (llm_tok_path) env->ReleaseStringUTFChars(llm_tokenizer_path, llm_tok_path);

    bool success = g_pipeline->init(config);

    if (!success) {
        LOGE("Pipeline initialization failed");
        g_pipeline.reset();
    } else {
        LOGI("Pipeline initialized successfully");
    }

    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_armm_translation_NativePipeline_nativeIsLLMActive(
    JNIEnv *env,
    jobject /* this */) {
    if (!g_pipeline) return JNI_FALSE;
    return g_pipeline->is_llm_active() ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_armm_translation_NativePipeline_nativeStart(
    JNIEnv *env,
    jobject /* this */) {
    
    if (!g_pipeline) {
        LOGE("Cannot start: pipeline not initialized");
        return JNI_FALSE;
    }
    
    LOGI("Starting pipeline...");
    bool success = g_pipeline->start();
    if (success) {
        LOGI("Pipeline started successfully");
    } else {
        LOGE("Pipeline start failed");
    }
    
    return success ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_armm_translation_NativePipeline_nativeStop(
    JNIEnv *env,
    jobject /* this */) {
    
    if (g_pipeline) {
        g_pipeline->stop();
        g_pipeline.reset();
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_armm_translation_NativePipeline_nativePushAudio(
    JNIEnv *env,
    jobject /* this */,
    jfloatArray audio_samples) {
    
    if (!g_pipeline) {
        return;
    }
    
    jsize len = env->GetArrayLength(audio_samples);
    jfloat* samples = env->GetFloatArrayElements(audio_samples, nullptr);
    
    if (samples) {
        static int push_count = 0;
        push_count++;
        if (push_count <= 5 || push_count % 100 == 0) {
            // Log first 5 pushes and every 100th thereafter
            float max_val = 0;
            for (jsize i = 0; i < len; i++) {
                float v = samples[i] < 0 ? -samples[i] : samples[i];
                if (v > max_val) max_val = v;
            }
            LOGD("pushAudio #%d: %d samples, peak=%.6f", push_count, len, max_val);
        }
        g_pipeline->push_audio(samples, len);
        env->ReleaseFloatArrayElements(audio_samples, samples, JNI_ABORT);
    }
}

extern "C" JNIEXPORT jfloatArray JNICALL
Java_com_armm_translation_NativePipeline_nativePopAudio(
    JNIEnv *env,
    jobject /* this */) {
    
    if (!g_pipeline) {
        return nullptr;
    }
    
    std::vector<float> audio_samples;
    if (!g_pipeline->pop_audio(audio_samples)) {
        return nullptr;
    }
    
    jfloatArray result = env->NewFloatArray(audio_samples.size());
    if (result) {
        env->SetFloatArrayRegion(result, 0, audio_samples.size(), 
                                audio_samples.data());
    }
    
    return result;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_armm_translation_NativePipeline_nativeGetCurrentEnglish(
    JNIEnv *env,
    jobject /* this */) {
    
    if (!g_pipeline) {
        return env->NewStringUTF("");
    }
    
    std::string text = g_pipeline->get_current_english();
    return env->NewStringUTF(text.c_str());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_armm_translation_NativePipeline_nativeGetCurrentHindi(
    JNIEnv *env,
    jobject /* this */) {

    if (!g_pipeline) {
        return env->NewStringUTF("");
    }

    std::string text = g_pipeline->get_current_hindi();
    return env->NewStringUTF(text.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_armm_translation_NativePipeline_nativeSetTranslationMode(
    JNIEnv *env,
    jobject /* this */,
    jint mode) {
    if (g_pipeline) {
        g_pipeline->set_translation_mode(static_cast<int>(mode));
        LOGI("Translation mode set to %d", static_cast<int>(mode));
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_armm_translation_NativePipeline_nativeGetTranslationMode(
    JNIEnv *env,
    jobject /* this */) {
    if (!g_pipeline) {
        return env->NewStringUTF("Unknown");
    }
    std::string mode = g_pipeline->get_translation_mode_name();
    return env->NewStringUTF(mode.c_str());
}

extern "C" JNIEXPORT jdouble JNICALL
Java_com_armm_translation_NativePipeline_nativeGetAcceptanceRate(
    JNIEnv *env,
    jobject /* this */) {
    if (!g_pipeline) return 0.0;
    return static_cast<jdouble>(g_pipeline->get_mt_acceptance_rate());
}
