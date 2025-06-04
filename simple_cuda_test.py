import torch
print("✓ PyTorch imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA devices: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
    
    # Test creating a CUDA tensor
    try:
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"✓ CUDA tensor created: {test_tensor}")
        print(f"Tensor device: {test_tensor.device}")
    except Exception as e:
        print(f"✗ Error creating CUDA tensor: {e}")
else:
    print("CUDA is not available - this may be expected if:")
    print("- No NVIDIA GPU is installed")
    print("- NVIDIA drivers are not installed") 
    print("- CUDA toolkit is not properly installed")
    print("- CUDA drivers are incompatible with PyTorch version")
