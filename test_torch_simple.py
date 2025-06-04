#!/usr/bin/env python3
"""Simple PyTorch test."""

try:
    import torch
    print("✅ PyTorch imported successfully")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
except Exception as e:
    print(f"❌ PyTorch import failed: {e}")
    import traceback
    traceback.print_exc()
