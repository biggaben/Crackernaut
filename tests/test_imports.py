#!/usr/bin/env python3
"""Test script to verify imports are working correctly."""

import sys
import os

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    print("Testing imports...")
    from src.utils.variant_utils import generate_variants
    print("✅ variant_utils import successful")
    
    from src.utils.config_utils import PROJECT_ROOT, load_configuration
    print("✅ config_utils import successful")
    print(f"Project root: {PROJECT_ROOT}")
    
    import os
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    print(f"Config file path: {config_path}")
    print(f"Config file exists: {os.path.exists(config_path)}")
    
    print("All imports working correctly!")
    
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback
    traceback.print_exc()
