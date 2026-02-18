/**
 * Host (desktop) inference entry point
 * Runs the speech-to-speech translation pipeline on host CPU
 */

#include "../pipeline/pipeline.h"
#include <iostream>
#include <fstream>
#include <string>
#include <thread>
#include <atomic>
#include <cstdint>
#include <vector>
#include <signal.h>

#ifdef USE_PORTAUDIO
#include <portaudio.h>
#endif

static std::atomic<bool> g_running{true};

void signal_handler(int signal) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
}

#ifdef USE_PORTAUDIO
// PortAudio callback for audio capture
int audio_input_callback(const void* inputBuffer,
                         void* outputBuffer,
                         unsigned long framesPerBuffer,
                         const PaStreamCallbackTimeInfo* timeInfo,
                         PaStreamCallbackFlags statusFlags,
                         void* userData) {
    Pipeline* pipeline = static_cast<Pipeline*>(userData);
    
    if (g_running.load() && inputBuffer != nullptr) {
        const float* samples = static_cast<const float*>(inputBuffer);
        pipeline->push_audio(samples, framesPerBuffer);
    }
    
    return g_running.load() ? paContinue : paComplete;
}

// PortAudio callback for audio playback
int audio_output_callback(const void* inputBuffer,
                          void* outputBuffer,
                          unsigned long framesPerBuffer,
                          const PaStreamCallbackTimeInfo* timeInfo,
                          PaStreamCallbackFlags statusFlags,
                          void* userData) {
    Pipeline* pipeline = static_cast<Pipeline*>(userData);
    float* samples = static_cast<float*>(outputBuffer);
    
    std::vector<float> audio_chunk;
    if (pipeline->pop_audio(audio_chunk)) {
        size_t copy_size = std::min(audio_chunk.size(), static_cast<size_t>(framesPerBuffer));
        std::copy(audio_chunk.begin(), audio_chunk.begin() + copy_size, samples);
        
        // Zero out remaining samples
        if (copy_size < framesPerBuffer) {
            std::fill(samples + copy_size, samples + framesPerBuffer, 0.0f);
        }
    } else {
        // No audio available, output silence
        std::fill(samples, samples + framesPerBuffer, 0.0f);
    }
    
    return g_running.load() ? paContinue : paComplete;
}

void setup_portaudio(Pipeline& pipeline) {
    PaError err = Pa_Initialize();
    if (err != paNoError) {
        std::cerr << "PortAudio initialization failed: " << Pa_GetErrorText(err) << std::endl;
        return;
    }
    
    const int SAMPLE_RATE = 16000;
    const int FRAMES_PER_BUFFER = 512;
    
    // Open input stream (microphone)
    PaStream* inputStream;
    err = Pa_OpenDefaultStream(&inputStream,
                                1,      // Input channels (mono)
                                0,      // Output channels
                                paFloat32,
                                SAMPLE_RATE,
                                FRAMES_PER_BUFFER,
                                audio_input_callback,
                                &pipeline);
    
    if (err != paNoError) {
        std::cerr << "Failed to open input stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_Terminate();
        return;
    }
    
    // Open output stream (speakers)
    PaStream* outputStream;
    err = Pa_OpenDefaultStream(&outputStream,
                                0,      // Input channels
                                1,      // Output channels (mono)
                                paFloat32,
                                SAMPLE_RATE,
                                FRAMES_PER_BUFFER,
                                audio_output_callback,
                                &pipeline);
    
    if (err != paNoError) {
        std::cerr << "Failed to open output stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(inputStream);
        Pa_Terminate();
        return;
    }
    
    // Start streams
    err = Pa_StartStream(inputStream);
    if (err != paNoError) {
        std::cerr << "Failed to start input stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_CloseStream(inputStream);
        Pa_CloseStream(outputStream);
        Pa_Terminate();
        return;
    }
    
    err = Pa_StartStream(outputStream);
    if (err != paNoError) {
        std::cerr << "Failed to start output stream: " << Pa_GetErrorText(err) << std::endl;
        Pa_StopStream(inputStream);
        Pa_CloseStream(inputStream);
        Pa_CloseStream(outputStream);
        Pa_Terminate();
        return;
    }
    
    std::cout << "Audio streams started. Press Ctrl+C to stop." << std::endl;
    
    // Keep running until interrupted
    while (g_running.load()) {
        Pa_Sleep(100);
        
        // Print current text
        std::string english = pipeline.get_current_english();
        std::string hindi = pipeline.get_current_hindi();
        
        if (!english.empty()) {
            std::cout << "\rEnglish: " << english << "          ";
            std::cout.flush();
        }
        if (!hindi.empty()) {
            std::cout << "\nHindi: " << hindi << std::endl;
        }
    }
    
    // Stop and close streams
    Pa_StopStream(inputStream);
    Pa_StopStream(outputStream);
    Pa_CloseStream(inputStream);
    Pa_CloseStream(outputStream);
    Pa_Terminate();
}
#endif

// Write 16-bit mono WAV from float samples in [-1, 1]
static bool write_wav_file(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    if (samples.empty()) return false;
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    const uint32_t num_samples = static_cast<uint32_t>(samples.size());
    const uint32_t data_bytes = num_samples * 2u;
    const uint32_t riff_size = 4 + 8 + 16 + 8 + data_bytes; // WAVE + fmt chunk + data chunk
    char riff[4] = { 'R', 'I', 'F', 'F' };
    char wave[4] = { 'W', 'A', 'V', 'E' };
    char fmt_id[4] = { 'f', 'm', 't', ' ' };
    const uint32_t fmt_size = 16u;
    const uint16_t audio_format = 1;   // PCM
    const uint16_t num_channels = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate = static_cast<uint32_t>(sample_rate) * num_channels * (bits_per_sample / 8);
    const uint16_t block_align = num_channels * (bits_per_sample / 8);
    char data_id[4] = { 'd', 'a', 't', 'a' };
    out.write(riff, 4);
    out.write(reinterpret_cast<const char*>(&riff_size), 4);
    out.write(wave, 4);
    out.write(fmt_id, 4);
    out.write(reinterpret_cast<const char*>(&fmt_size), 4);
    out.write(reinterpret_cast<const char*>(&audio_format), 2);
    out.write(reinterpret_cast<const char*>(&num_channels), 2);
    out.write(reinterpret_cast<const char*>(&sample_rate), 4);
    out.write(reinterpret_cast<const char*>(&byte_rate), 4);
    out.write(reinterpret_cast<const char*>(&block_align), 2);
    out.write(reinterpret_cast<const char*>(&bits_per_sample), 2);
    out.write(data_id, 4);
    out.write(reinterpret_cast<const char*>(&data_bytes), 4);
    for (float s : samples) {
        int16_t v = static_cast<int16_t>(s * 32767.0f);
        if (v > 32767) v = 32767;
        if (v < -32768) v = -32768;
        out.write(reinterpret_cast<const char*>(&v), 2);
    }
    return out.good();
}

// Parse WAV: find "data" chunk and optional "fmt " for sample rate.
// Returns data_start_offset in bytes, or -1 on error. *out_sample_rate = 0 if unknown.
static long find_wav_data_offset(std::istream& in, int* out_sample_rate) {
    char buf[12];
    if (!in.read(buf, 12) || buf[0] != 'R' || buf[1] != 'I' || buf[2] != 'F' || buf[3] != 'F'
        || buf[8] != 'W' || buf[9] != 'A' || buf[10] != 'V' || buf[11] != 'E') {
        return -1;
    }
    if (out_sample_rate) *out_sample_rate = 16000;
    while (in) {
        char id[4];
        uint32_t chunk_size;
        if (!in.read(id, 4) || !in.read(reinterpret_cast<char*>(&chunk_size), 4)) break;
        if (id[0] == 'f' && id[1] == 'm' && id[2] == 't' && id[3] == ' ') {
            uint16_t fmt_code, n_channels;
            uint32_t sample_rate;
            if (chunk_size >= 16 && in.read(reinterpret_cast<char*>(&fmt_code), 2)
                && in.read(reinterpret_cast<char*>(&n_channels), 2)
                && in.read(reinterpret_cast<char*>(&sample_rate), 4)) {
                if (out_sample_rate) *out_sample_rate = static_cast<int>(sample_rate);
            }
            in.seekg(static_cast<std::streamoff>(chunk_size - 8), std::ios::cur);
            continue;
        }
        if (id[0] == 'd' && id[1] == 'a' && id[2] == 't' && id[3] == 'a') {
            return static_cast<long>(in.tellg());
        }
        in.seekg(static_cast<std::streamoff>(chunk_size), std::ios::cur);
    }
    return -1;
}

// File-based audio I/O (fallback, no PortAudio)
void process_audio_files(Pipeline& pipeline, 
                        const std::string& input_wav,
                        const std::string& output_wav) {
    std::cout << "File-based processing mode" << std::endl;
    std::cout << "Reading from: " << input_wav << std::endl;
    std::cout << "Writing to: " << output_wav << std::endl;
    
    std::ifstream input_file(input_wav, std::ios::binary);
    if (!input_file.is_open()) {
        std::cerr << "Failed to open input file: " << input_wav << std::endl;
        return;
    }
    int file_sample_rate = 0;
    long data_offset = find_wav_data_offset(input_file, &file_sample_rate);
    if (data_offset < 0) {
        std::cerr << "Error: Not a valid WAV file (could not find data chunk)" << std::endl;
        input_file.close();
        return;
    }
    input_file.seekg(data_offset);
    if (file_sample_rate != 0 && file_sample_rate != 16000) {
        std::cerr << "Warning: WAV sample rate is " << file_sample_rate << " Hz; pipeline expects 16 kHz. Resample the file or results may be wrong." << std::endl;
    }
    
    const int SAMPLE_RATE = 16000;
    const int CHUNK_SIZE = SAMPLE_RATE / 10; // 100 ms chunks
    std::vector<int16_t> int16_buffer(CHUNK_SIZE);
    std::vector<float> float_buffer(CHUNK_SIZE);
    std::vector<float> output_accum;

    while (g_running.load() && input_file.good()) {
        input_file.read(reinterpret_cast<char*>(int16_buffer.data()), 
                       CHUNK_SIZE * sizeof(int16_t));
        
        size_t samples_read = input_file.gcount() / sizeof(int16_t);
        
        if (samples_read == 0) break;
        
        // Convert int16 to float32
        for (size_t i = 0; i < samples_read; i++) {
            float_buffer[i] = int16_buffer[i] / 32768.0f;
        }
        
        // Push to pipeline (accumulate in ASR wrapper)
        pipeline.push_audio(float_buffer.data(), samples_read);
        
        // Pop synthesized audio and accumulate for output WAV
        std::vector<float> output_audio;
        while (pipeline.pop_audio(output_audio)) {
            output_accum.insert(output_accum.end(), output_audio.begin(), output_audio.end());
        }
        
        // Print status
        std::string english = pipeline.get_current_english();
        std::string hindi = pipeline.get_current_hindi();
        
        if (!english.empty() && english != "[BLANK_AUDIO]") {
            std::cout << "\rEnglish: " << english << "          ";
            std::cout.flush();
        }
        if (!hindi.empty()) {
            std::cout << "\nHindi: " << hindi << std::endl;
        }
    }
    
    input_file.close();
    std::cout << "\nFile read complete. Waiting for ASR to drain..." << std::endl;

    // Wait for audio queue to drain (ASR thread processes all pushed audio)
    pipeline.wait_audio_drained(5000);

    std::cout << "Flushing ASR..." << std::endl;

    // Flush ASR to process remaining audio
    pipeline.flush_asr();
    
    // Wait for pipeline to process (up to 15 seconds for LLM), keep collecting output audio
    for (int i = 0; i < 150; i++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        std::vector<float> output_audio;
        while (pipeline.pop_audio(output_audio)) {
            output_accum.insert(output_accum.end(), output_audio.begin(), output_audio.end());
        }
        std::string english = pipeline.get_current_english();
        std::string hindi = pipeline.get_current_hindi();
        if (!english.empty() && english != "[BLANK_AUDIO]") {
            std::cout << "\rEnglish: " << english << "          ";
            std::cout.flush();
        }
        if (!hindi.empty()) {
            std::cout << "\nHindi: " << hindi << std::endl;
        }
    }

    // Write output WAV
    if (!output_wav.empty()) {
        if (output_accum.empty()) {
            std::cout << "\nNo output audio to write (no TTS output). Check that translation produced Hindi text." << std::endl;
        } else if (write_wav_file(output_wav, output_accum, SAMPLE_RATE)) {
            std::cout << "\nOutput audio written: " << output_wav << " (" << output_accum.size() << " samples)" << std::endl;
        } else {
            std::cerr << "Failed to write output file: " << output_wav << " (check path and permissions)" << std::endl;
        }
    }
    
    // Final status
    std::string english = pipeline.get_current_english();
    std::string hindi = pipeline.get_current_hindi();
    
    std::cout << "\n\n=== Final Results ===" << std::endl;
    if (!english.empty() && english != "[BLANK_AUDIO]") {
        std::cout << "English: " << english << std::endl;
    } else {
        std::cout << "English: (no speech detected)" << std::endl;
    }
    if (!hindi.empty()) {
        std::cout << "Hindi: " << hindi << std::endl;
    } else {
        std::cout << "Hindi: (no translation)" << std::endl;
    }
    std::cout << "\nProcessing complete." << std::endl;
}

int main(int argc, char* argv[]) {
    // Setup signal handler
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "=== Speech-to-Speech Translation (Desktop CPU) ===" << std::endl;
    
    // Parse command line arguments
    std::string asr_model_path;
    std::string mt_model_path;
    std::string tts_model_path;
    std::string llm_model_path;
    std::string llm_tokenizer_path;
    std::string translation_mode_str;
    std::string input_wav;
    std::string output_wav;
    bool use_files = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--asr-model" && i + 1 < argc) {
            asr_model_path = argv[++i];
        } else if (arg == "--mt-model" && i + 1 < argc) {
            mt_model_path = argv[++i];
        } else if (arg == "--tts-model" && i + 1 < argc) {
            tts_model_path = argv[++i];
        } else if (arg == "--llm-model" && i + 1 < argc) {
            llm_model_path = argv[++i];
        } else if (arg == "--llm-tokenizer" && i + 1 < argc) {
            llm_tokenizer_path = argv[++i];
        } else if (arg == "--translation-mode" && i + 1 < argc) {
            translation_mode_str = argv[++i];
        } else if (arg == "--input" && i + 1 < argc) {
            input_wav = argv[++i];
            use_files = true;
        } else if (arg == "--output" && i + 1 < argc) {
            output_wav = argv[++i];
            use_files = true;
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  --asr-model PATH      Path to ASR model (required)" << std::endl;
            std::cout << "  --mt-model PATH       Path to MT model (required)" << std::endl;
            std::cout << "  --tts-model PATH      Path to TTS model (required)" << std::endl;
            std::cout << "  --llm-model PATH      Path to LLM model (.pte file, optional)" << std::endl;
            std::cout << "  --llm-tokenizer PATH  Path to LLM tokenizer (tokenizer.json, optional)" << std::endl;
            std::cout << "  --translation-mode M  Translation mode: speed|balanced|quality (default: balanced)" << std::endl;
            std::cout << "  --input PATH          Input WAV file (file mode)" << std::endl;
            std::cout << "  --output PATH         Output WAV file (file mode)" << std::endl;
            std::cout << "  --help                Show this help message" << std::endl;
            return 0;
        }
    }
    
    // Check required arguments
    if (asr_model_path.empty() || mt_model_path.empty() || tts_model_path.empty()) {
        std::cerr << "Error: Model paths required. Use --help for usage." << std::endl;
        return 1;
    }
    
    // Initialize pipeline
    Pipeline pipeline;
    PipelineConfig config;
    config.asr_model_path = asr_model_path;
    config.mt_model_path = mt_model_path;
    config.tts_model_path = tts_model_path;
    config.llm_model_path = llm_model_path;
    config.llm_tokenizer_path = llm_tokenizer_path;
    if (translation_mode_str == "speed") config.translation_mode = 0;
    else if (translation_mode_str == "quality") config.translation_mode = 2;
    else config.translation_mode = 1; // balanced (default)
    config.sample_rate = 16000;
    config.chunk_size_ms = 80;
    
    std::cout << "Initializing pipeline..." << std::endl;
    if (!pipeline.init(config)) {
        std::cerr << "Failed to initialize pipeline" << std::endl;
        return 1;
    }
    
    std::cout << "Starting pipeline..." << std::endl;
    if (!pipeline.start()) {
        std::cerr << "Failed to start pipeline" << std::endl;
        return 1;
    }

    std::cout << "Translation mode: " << pipeline.get_translation_mode_name() << std::endl;
    if (pipeline.is_llm_active()) {
        std::cout << "LLM backend: active (Qwen3-0.6B via ExecuTorch)" << std::endl;
    } else {
        std::cout << "LLM backend: inactive (using Marian NMT)" << std::endl;
    }

    // Run audio processing
#ifdef USE_PORTAUDIO
    if (!use_files) {
        setup_portaudio(pipeline);
    } else {
        process_audio_files(pipeline, input_wav, output_wav);
    }
#else
    if (use_files && !input_wav.empty()) {
        process_audio_files(pipeline, input_wav, output_wav);
    } else {
        std::cerr << "Error: PortAudio not available. Use --input and --output for file mode." << std::endl;
        return 1;
    }
#endif
    
    // Print speculative decoding stats if applicable
    double acceptance = pipeline.get_mt_acceptance_rate();
    if (acceptance > 0.0) {
        std::cout << "Speculative acceptance rate: "
                  << static_cast<int>(acceptance * 100) << "%" << std::endl;
    }

    // Stop pipeline
    pipeline.stop();

    std::cout << "Pipeline stopped." << std::endl;
    return 0;
}
