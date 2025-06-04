import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

print("Attempting to import torch...")
try:
    import torch
    print("✓ PyTorch imported successfully!")
    print(f"PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"✗ ImportError: {e}")
except Exception as e:
    print(f"✗ Other error: {e}")
    import traceback
    traceback.print_exc()

print("Script completed.")
