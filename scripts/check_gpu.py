#!/usr/bin/env python3
"""
GPU Status Checker for Crackernaut
Displays CUDA availability and device information for ML development.
"""

import sys


def check_gpu_status() -> None:
    """Check and display GPU/CUDA status information."""
    try:
        import torch

        print("=" * 50)
        print("🔍 Crackernaut GPU Status Check")
        print("=" * 50)

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"CUDA Available: {'✅ YES' if cuda_available else '❌ NO'}")

        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"CUDA Devices: {device_count}")
            print()

            # List all available devices
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_total = props.total_memory
                memory_gb = memory_total / (1024**3)

                print(f"Device {i}: {device_name}")
                print(f"  Memory: {memory_gb:.1f} GB")
                print(f"  Compute Capability: {props.major}.{props.minor}")
                print()

            # Current device
            current_device = torch.cuda.current_device()
            print(f"Current Device: {current_device}")

            # Memory usage
            if device_count > 0:
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                print(f"Memory Allocated: {memory_allocated:.2f} GB")
                print(f"Memory Cached: {memory_cached:.2f} GB")
        else:
            print("\n⚠️  CUDA not available. Running on CPU only.")
            print("   For GPU acceleration, ensure:")
            print("   1. NVIDIA GPU is installed")
            print("   2. CUDA drivers are installed")
            print("   3. PyTorch CUDA version is installed")
            print("\n   Install CUDA PyTorch with:")
            print("   uv sync --extra cuda")

        print("\n" + "=" * 50)

    except ImportError as e:
        print(f"❌ Error importing PyTorch: {e}")
        print("Install PyTorch with: uv sync")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error checking GPU status: {e}")
        sys.exit(1)


if __name__ == "__main__":
    check_gpu_status()
