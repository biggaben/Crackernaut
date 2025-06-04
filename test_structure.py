#!/usr/bin/env python3
"""Quick test to verify the project structure reorganization worked."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_file_structure():
    """Test that all expected files are in their new locations."""
    
    print("Testing project structure reorganization...")
    
    # Check tests directory
    tests_files = [
        "tests/test_variants.py",
        "tests/test_crackernaut.py", 
        "tests/test_imports.py",
        "tests/test_torch.py",
        "tests/test_basic_functionality.py",
        "tests/__init__.py"
    ]
    
    # Check docs directory
    docs_files = [
        "docs/README.md",
        "docs/SETUP.md",
        "docs/STRUCTURE.md",
        "docs/AGENTS.md",
        "docs/MARKDOWN_STYLE_GUIDE.md"
    ]
    
    # Check scripts/dev directory
    dev_files = [
        "scripts/dev/check_cuda.py",
        "scripts/dev/debug_torch.py",
        "scripts/dev/simple_cuda_test.py",
        "scripts/dev/simple_torch_test.py",
        "scripts/dev/README.md"
    ]
    
    all_files = tests_files + docs_files + dev_files
    
    missing_files = []
    existing_files = []
    
    for file_path in all_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    print("\nResults:")
    print(f"✅ Existing files: {len(existing_files)}")
    print(f"❌ Missing files: {len(missing_files)}")
    
    if missing_files:
        print("\nMissing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("\n🎉 All files are in their expected locations!")
        return True

def test_old_files_removed():
    """Test that old files were properly moved/removed."""
    
    print("\nTesting that old files were removed...")
    
    old_files_that_should_not_exist = [
        "test_variants.py",
        "test_crackernaut.py", 
        "test_imports.py",
        "test_torch.py",
        "test_basic_functionality.py",
        "check_cuda.py",
        "debug_torch.py",
        "simple_cuda_test.py",
        "simple_torch_test.py",
        "COPILOT_SETUP.md",
        "STRUCTURE.md",
        "AGENTS.md"
    ]
    
    still_existing = []
    properly_removed = []
    
    for file_path in old_files_that_should_not_exist:
        if os.path.exists(file_path):
            still_existing.append(file_path)
            print(f"⚠️  {file_path} still exists in root")
        else:
            properly_removed.append(file_path)
            print(f"✅ {file_path} properly moved/removed")
    
    print("\nResults:")
    print(f"✅ Properly moved/removed: {len(properly_removed)}")
    print(f"⚠️  Still in root: {len(still_existing)}")
    
    return len(still_existing) == 0

if __name__ == "__main__":
    print("=" * 60)
    print("CRACKERNAUT PROJECT STRUCTURE VERIFICATION")
    print("=" * 60)
    
    structure_ok = test_file_structure()
    cleanup_ok = test_old_files_removed()
    
    print("\n" + "=" * 60)
    if structure_ok and cleanup_ok:
        print("🎉 PROJECT REORGANIZATION COMPLETED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("❌ Some issues found with project reorganization")
        sys.exit(1)
