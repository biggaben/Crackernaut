import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
