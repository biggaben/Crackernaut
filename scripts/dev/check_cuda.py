#!/usr/bin/env python3
"""
Quick CUDA availability check for PyTorch installation.
"""

import torch

def main():
    print("=" * 50)
    print("PyTorch CUDA Installation Check")
    print("=" * 50)
    
    # PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:        # CUDA compilation info
        print("CUDA compiled with PyTorch: Yes")
        
        # Number of CUDA devices
        device_count = torch.cuda.device_count()
        print(f"CUDA Devices: {device_count}")
        
        # Device details
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"Device {i}: {device_name}")
            
        # Current device
        current_device = torch.cuda.current_device()
        print(f"Current CUDA Device: {current_device}")
        
        # Memory info
        if device_count > 0:
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"Memory Allocated: {memory_allocated:.2f} GB")
            print(f"Memory Reserved: {memory_reserved:.2f} GB")
    else:
        print("CUDA is not available. Reasons could include:")
        print("- No NVIDIA GPU installed")
        print("- NVIDIA drivers not installed")
        print("- CUDA toolkit not installed")
        print("- PyTorch CPU-only version installed")
    
    # Test tensor creation
    try:
        if cuda_available:
            print("\nTesting CUDA tensor creation...")
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            print(f"CUDA tensor created successfully: {x}")
            print(f"Tensor device: {x.device}")
        else:
            print("\nTesting CPU tensor creation...")
            x = torch.tensor([1.0, 2.0, 3.0])
            print(f"CPU tensor created successfully: {x}")
            print(f"Tensor device: {x.device}")
    except Exception as e:
        print(f"Error creating tensor: {e}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
