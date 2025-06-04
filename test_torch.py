#!/usr/bin/env python3
"""Test PyTorch CUDA functionality."""

try:
    import torch
    print(f"✓ PyTorch imported successfully")
    print(f"  Version: {torch.__version__}")
    
    # Test CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"  CUDA version: {torch.version.cuda}")
        device_count = torch.cuda.device_count()
        print(f"  GPU count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"    GPU {i}: {device_name}")
            
        # Test basic CUDA operation
        if device_count > 0:
            x = torch.tensor([1.0, 2.0, 3.0]).cuda()
            y = x * 2
            print(f"  CUDA tensor test: {y.cpu().tolist()}")
    else:
        print("  CUDA not available - using CPU only")
        
except ImportError as e:
    print(f"✗ Failed to import PyTorch: {e}")
except Exception as e:
    print(f"✗ Error testing PyTorch: {e}")
