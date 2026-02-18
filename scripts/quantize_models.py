#!/usr/bin/env python3
"""
Quantize ONNX models to INT8 for mobile deployment.
Reduces model size ~75% and improves ARM NEON inference speed.

Usage:
    python scripts/quantize_models.py
"""

import os
import sys

def quantize_model(input_path, output_path):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print(f"  {os.path.basename(input_path)} → {os.path.basename(output_path)}")
    quantize_dynamic(input_path, output_path, weight_type=QuantType.QInt8)
    in_size = os.path.getsize(input_path) / (1024 * 1024)
    out_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"    {in_size:.0f} MB → {out_size:.0f} MB ({out_size/in_size*100:.0f}%)")

def main():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

    mt_dir = os.path.join(models_dir, "mt", "onnx")
    tts_dir = os.path.join(models_dir, "tts")

    print("=== ONNX Model Quantization (INT8) ===\n")

    # MT models
    print("MT models:")
    for name in ["encoder_model.onnx", "decoder_model.onnx", "decoder_with_past_model.onnx"]:
        src = os.path.join(mt_dir, name)
        dst = os.path.join(mt_dir, name.replace(".onnx", "_int8.onnx"))
        if os.path.exists(src):
            quantize_model(src, dst)
        else:
            print(f"  SKIP {name} (not found)")

    # TTS model
    print("\nTTS model:")
    tts_src = os.path.join(tts_dir, "hi_IN-rohan-medium.onnx")
    tts_dst = os.path.join(tts_dir, "hi_IN-rohan-medium_int8.onnx")
    if os.path.exists(tts_src):
        quantize_model(tts_src, tts_dst)
    else:
        print(f"  SKIP hi_IN-rohan-medium.onnx (not found)")

    print("\nDone. Use *_int8.onnx models for mobile deployment.")

if __name__ == "__main__":
    main()
