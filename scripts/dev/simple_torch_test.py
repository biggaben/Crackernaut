import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():        # Get CUDA runtime version if available
        try:
            import torch.backends.cudnn as cudnn
            print(f"cuDNN enabled: {cudnn.enabled}")
            print(f"cuDNN version: {cudnn.version()}")
        except Exception as e:
            print(f"cuDNN info not available: {e}")
        
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        # Get GPU memory info
        print(f"GPU memory allocated: {torch.cuda.memory_allocated()} bytes")
        print(f"GPU memory cached: {torch.cuda.memory_reserved()} bytes")
        
        # Test basic CUDA tensor operation
        device = torch.device('cuda:0')
        test_tensor = torch.tensor([1.0, 2.0, 3.0]).to(device)
        print(f"Test CUDA tensor: {test_tensor}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
