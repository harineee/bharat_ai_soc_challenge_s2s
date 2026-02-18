#!/bin/bash
# ============================================================
# Setup Android native dependencies for ARMM
# Downloads: ONNX Runtime (native C++) + SentencePiece (static)
# Target: arm64-v8a only
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LIBS_DIR="$PROJECT_ROOT/libs"
TMP_DIR="$PROJECT_ROOT/.tmp_android_deps"

ORT_VERSION="1.16.3"
SP_VERSION="0.2.0"

echo "=== Setting up Android native dependencies ==="
echo "Project root: $PROJECT_ROOT"
echo "ONNX Runtime: v${ORT_VERSION}"
echo ""

mkdir -p "$LIBS_DIR" "$TMP_DIR"

# -------------------------------------------------------
# 1. ONNX Runtime — headers + arm64-v8a shared library
# -------------------------------------------------------
ORT_OUT="$LIBS_DIR/onnxruntime"
if [ -f "$ORT_OUT/jni/arm64-v8a/libonnxruntime.so" ] && [ -d "$ORT_OUT/headers/onnxruntime" ]; then
    echo "[ORT] Already present — skipping download"
else
    echo "[ORT] Downloading ONNX Runtime ${ORT_VERSION} for Android..."

    # Download the AAR (contains arm64-v8a .so)
    AAR_URL="https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"
    AAR_FILE="$TMP_DIR/onnxruntime-android-${ORT_VERSION}.aar"

    if [ ! -f "$AAR_FILE" ]; then
        wget -q --show-progress -O "$AAR_FILE" "$AAR_URL"
    fi

    # Extract .so from AAR (AAR is a ZIP file)
    mkdir -p "$ORT_OUT/jni/arm64-v8a"
    cd "$TMP_DIR"

    # Try unzip first, fall back to Python zipfile
    if command -v unzip >/dev/null 2>&1; then
        unzip -qo "$AAR_FILE" "jni/arm64-v8a/*" -d ort_aar 2>/dev/null || true
        [ -f "ort_aar/jni/arm64-v8a/libonnxruntime.so" ] && cp ort_aar/jni/arm64-v8a/*.so "$ORT_OUT/jni/arm64-v8a/"
    fi

    # Fallback: use Python to extract
    if [ ! -f "$ORT_OUT/jni/arm64-v8a/libonnxruntime.so" ]; then
        python3 -c "
import zipfile, os
aar = '$AAR_FILE'
out = '$ORT_OUT/jni/arm64-v8a'
os.makedirs(out, exist_ok=True)
with zipfile.ZipFile(aar, 'r') as z:
    for name in z.namelist():
        if 'jni/arm64-v8a/' in name and name.endswith('.so'):
            data = z.read(name)
            dest = os.path.join(out, os.path.basename(name))
            with open(dest, 'wb') as f:
                f.write(data)
            print(f'  Extracted: {os.path.basename(name)} ({len(data)} bytes)')
" 2>/dev/null || true
    fi

    if [ -f "$ORT_OUT/jni/arm64-v8a/libonnxruntime.so" ]; then
        echo "[ORT] Extracted libonnxruntime.so for arm64-v8a"
    else
        echo "[ORT] ERROR: Could not extract .so from AAR"
    fi

    # Download headers from the Linux/generic release (headers are platform-independent)
    HEADERS_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz"
    HEADERS_FILE="$TMP_DIR/onnxruntime-linux-x64-${ORT_VERSION}.tgz"

    if [ ! -f "$HEADERS_FILE" ]; then
        echo "[ORT] Downloading headers..."
        wget -q --show-progress -O "$HEADERS_FILE" "$HEADERS_URL" 2>/dev/null || true
    fi

    mkdir -p "$ORT_OUT/headers"
    if [ -f "$HEADERS_FILE" ]; then
        tar -xzf "$HEADERS_FILE" -C "$TMP_DIR/" 2>/dev/null || true
        EXTRACTED_DIR="$TMP_DIR/onnxruntime-linux-x64-${ORT_VERSION}"
        if [ -d "$EXTRACTED_DIR/include" ]; then
            # Copy flat headers
            cp -r "$EXTRACTED_DIR/include/"* "$ORT_OUT/headers/"

            # Also create nested structure for #include <onnxruntime/core/session/...>
            mkdir -p "$ORT_OUT/headers/onnxruntime/core/session"
            for hdr in "$EXTRACTED_DIR/include/"*.h; do
                [ -f "$hdr" ] && cp "$hdr" "$ORT_OUT/headers/onnxruntime/core/session/"
            done
            echo "[ORT] Headers installed (flat + nested onnxruntime/core/session/)"
        fi
    fi

    # If headers download failed, try downloading from ORT source repo
    if [ ! -f "$ORT_OUT/headers/onnxruntime_cxx_api.h" ] && [ ! -d "$ORT_OUT/headers/onnxruntime" ]; then
        echo "[ORT] Attempting to download headers from source repo..."
        ORT_HEADERS_BASE="https://raw.githubusercontent.com/microsoft/onnxruntime/v${ORT_VERSION}/include/onnxruntime/core/session"
        mkdir -p "$ORT_OUT/headers/onnxruntime/core/session"
        for hdr in onnxruntime_c_api.h onnxruntime_cxx_api.h onnxruntime_cxx_inline.h onnxruntime_float16.h onnxruntime_run_options_config_keys.h onnxruntime_session_options_config_keys.h; do
            wget -q -O "$ORT_OUT/headers/onnxruntime/core/session/$hdr" "$ORT_HEADERS_BASE/$hdr" 2>/dev/null || true
        done
        if [ -f "$ORT_OUT/headers/onnxruntime/core/session/onnxruntime_cxx_api.h" ]; then
            echo "[ORT] Headers downloaded from source repo"
        else
            echo "[ORT] WARNING: Could not download headers automatically."
            echo "[ORT] Manual step required: place ONNX Runtime headers in $ORT_OUT/headers/"
            echo "[ORT] Download from: https://github.com/microsoft/onnxruntime/releases"
        fi
    fi
fi

# Verify ORT
echo ""
if [ -f "$ORT_OUT/jni/arm64-v8a/libonnxruntime.so" ]; then
    ORT_SIZE=$(du -h "$ORT_OUT/jni/arm64-v8a/libonnxruntime.so" | cut -f1)
    echo "[ORT] OK: libonnxruntime.so ($ORT_SIZE)"
else
    echo "[ORT] MISSING: libonnxruntime.so"
fi
if [ -f "$ORT_OUT/headers/onnxruntime/core/session/onnxruntime_cxx_api.h" ]; then
    echo "[ORT] OK: headers present (onnxruntime/core/session/)"
elif [ -f "$ORT_OUT/headers/onnxruntime_cxx_api.h" ]; then
    echo "[ORT] OK: headers present (flat)"
else
    echo "[ORT] MISSING: headers"
fi

# -------------------------------------------------------
# 2. SentencePiece — cross-compile static library
# -------------------------------------------------------
SP_OUT="$LIBS_DIR/sentencepiece"
if [ -f "$SP_OUT/lib/arm64-v8a/libsentencepiece.a" ]; then
    echo ""
    echo "[SP] Already present — skipping build"
else
    echo ""
    echo "[SP] Building SentencePiece for Android arm64-v8a..."

    # Check for NDK
    NDK_PATH=""
    for candidate in \
        "$ANDROID_NDK_HOME" \
        "$ANDROID_NDK" \
        "$HOME/Android/Sdk/ndk/"*/ \
        "/opt/android-ndk/"* \
        "/usr/local/lib/android/sdk/ndk/"*/; do
        if [ -f "${candidate}build/cmake/android.toolchain.cmake" ] 2>/dev/null; then
            NDK_PATH="$candidate"
            break
        fi
    done

    if [ -z "$NDK_PATH" ]; then
        echo "[SP] WARNING: Android NDK not found. SentencePiece must be cross-compiled manually."
        echo "[SP] Install NDK via Android Studio SDK Manager or:"
        echo "     sdkmanager --install 'ndk;25.2.9519653'"
        echo "[SP] Then set ANDROID_NDK_HOME and re-run this script."
        echo ""
        echo "[SP] Creating placeholder directory..."
        mkdir -p "$SP_OUT/include" "$SP_OUT/lib/arm64-v8a"

        # Download SentencePiece source for future compilation
        SP_SRC="$TMP_DIR/sentencepiece"
        if [ ! -d "$SP_SRC" ]; then
            echo "[SP] Downloading SentencePiece source..."
            git clone --depth 1 https://github.com/google/sentencepiece.git "$SP_SRC" 2>/dev/null || true
        fi
        if [ -d "$SP_SRC/src" ]; then
            cp "$SP_SRC/src/sentencepiece_processor.h" "$SP_OUT/include/" 2>/dev/null || true
            cp "$SP_SRC/src/sentencepiece_model.pb.h" "$SP_OUT/include/" 2>/dev/null || true
            echo "[SP] Headers copied. Library must be cross-compiled with NDK."
        fi
    else
        echo "[SP] Using NDK at: $NDK_PATH"
        TOOLCHAIN="$NDK_PATH/build/cmake/android.toolchain.cmake"

        SP_SRC="$TMP_DIR/sentencepiece"
        if [ ! -d "$SP_SRC" ]; then
            git clone --depth 1 https://github.com/google/sentencepiece.git "$SP_SRC"
        fi

        SP_BUILD="$TMP_DIR/sentencepiece-build-android"
        mkdir -p "$SP_BUILD"
        cd "$SP_BUILD"

        cmake "$SP_SRC" \
            -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_PLATFORM=android-26 \
            -DCMAKE_BUILD_TYPE=Release \
            -DSPM_ENABLE_SHARED=OFF \
            -DSPM_ENABLE_TCMALLOC=OFF \
            -DSPM_USE_BUILTIN_PROTOBUF=ON \
            -DCMAKE_INSTALL_PREFIX="$SP_OUT" \
            2>&1 | tail -5

        cmake --build . -j"$(nproc)" 2>&1 | tail -5

        # Install to output dir
        mkdir -p "$SP_OUT/include" "$SP_OUT/lib/arm64-v8a"
        cp -f src/libsentencepiece.a "$SP_OUT/lib/arm64-v8a/" 2>/dev/null || true
        cp -f "$SP_SRC/src/sentencepiece_processor.h" "$SP_OUT/include/"
        cp -f "$SP_SRC/src/sentencepiece_model.pb.h" "$SP_OUT/include/" 2>/dev/null || true

        echo "[SP] Build complete"
    fi
fi

# Verify SP
echo ""
if [ -f "$SP_OUT/lib/arm64-v8a/libsentencepiece.a" ]; then
    SP_SIZE=$(du -h "$SP_OUT/lib/arm64-v8a/libsentencepiece.a" | cut -f1)
    echo "[SP] OK: libsentencepiece.a ($SP_SIZE)"
else
    echo "[SP] MISSING: libsentencepiece.a (NDK required to build)"
fi
if [ -f "$SP_OUT/include/sentencepiece_processor.h" ]; then
    echo "[SP] OK: headers present"
else
    echo "[SP] MISSING: headers"
fi

# -------------------------------------------------------
# Cleanup
# -------------------------------------------------------
echo ""
echo "=== Dependency setup complete ==="
echo ""
echo "Directory layout:"
echo "  libs/"
echo "  ├── onnxruntime/"
echo "  │   ├── headers/          (C++ API headers)"
echo "  │   └── jni/arm64-v8a/   (libonnxruntime.so)"
echo "  └── sentencepiece/"
echo "      ├── include/          (sentencepiece_processor.h)"
echo "      └── lib/arm64-v8a/   (libsentencepiece.a)"
echo ""
echo "Next step: run ./scripts/prepare_models_android.sh"
