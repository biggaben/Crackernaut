#!/usr/bin/env python3
"""Simple test to verify crackernaut imports work."""

import sys
sys.path.insert(0, '.')

try:
    print("Testing crackernaut imports...")
    
    # Test individual imports
    from src.utils.config_utils import PROJECT_ROOT
    print(f"✅ config_utils: PROJECT_ROOT = {PROJECT_ROOT}")
    
    from src.utils.variant_utils import generate_variants
    print("✅ variant_utils: generate_variants imported")
    
    # Test generating a simple variant
    variants = generate_variants("test123", 20, 1)
    print(f"✅ Generated {len(variants)} variants for 'test123'")
    
    print("✅ All imports and basic functionality working!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
