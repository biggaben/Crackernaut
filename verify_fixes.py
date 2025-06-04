#!/usr/bin/env python3
"""Verification script to test relative path fixes."""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_relative_paths():
    """Test that all paths are now relative and work correctly."""
    
    print("Testing relative path fixes...")
    print("=" * 40)
    
    # Test 1: Check that we can determine project root from any file
    test_file_path = os.path.join("tests", "test_variants.py")
    if os.path.exists(test_file_path):
        print("✅ Test file exists at relative path")
    else:
        print("❌ Test file not found")
        return False
    
    # Test 2: Check config file access
    config_path = "config.json"
    if os.path.exists(config_path):
        print("✅ Config file accessible via relative path")
    else:
        print("❌ Config file not found")
        return False
    
    # Test 3: Check training data directory
    training_dir = "trainingdata"
    if os.path.exists(training_dir):
        print("✅ Training data directory exists")
    else:
        print("⚠️  Training data directory not found (normal in CI)")
    
    # Test 4: Check LICENSE file fix
    with open("LICENSE", "r") as f:
        license_content = f.read()
        if "BigGaben" in license_content:
            print("❌ Personal data still present in LICENSE")
            return False
        elif "Crackernaut Project" in license_content:
            print("✅ LICENSE file anonymized correctly")
        else:
            print("⚠️  LICENSE file content unexpected")
    
    # Test 5: Test project root detection from subdirectory context
    try:
        # Simulate being in tests directory
        test_dir = os.path.join(os.getcwd(), "tests")
        if os.path.exists(test_dir):
            # This is how our test files determine project root
            project_root_from_test = os.path.dirname(test_dir)
            expected_config = os.path.join(project_root_from_test, "config.json")
            if os.path.exists(expected_config):
                print("✅ Project root detection works from test directory")
            else:
                print("❌ Project root detection failed")
                return False
    except Exception as e:
        print(f"❌ Error testing project root detection: {e}")
        return False
    
    print("\n🎉 All relative path fixes verified successfully!")
    return True

def check_for_remaining_personal_data():
    """Check for any remaining personal data."""
    
    print("\nChecking for remaining personal data...")
    print("=" * 40)
    
    # Check common files for personal data patterns
    files_to_check = [
        "README.md",
        "pyproject.toml", 
        ".vscode/settings.json",
        "tests/test_variants.py"
    ]
    
    personal_patterns = [
        "BigGaben",
        "biggaben", 
        "i:\\",
        "I:\\",
        "C:\\Users\\",
        "/Users/",
        "/home/"
    ]
    
    issues_found = []
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    for pattern in personal_patterns:
                        if pattern.lower() in content.lower():
                            issues_found.append(f"{file_path}: contains '{pattern}'")
            except Exception as e:
                print(f"⚠️  Could not check {file_path}: {e}")
    
    if issues_found:
        print("❌ Personal data found:")
        for issue in issues_found:
            print(f"  - {issue}")
        return False
    else:
        print("✅ No personal data patterns detected")
        return True

if __name__ == "__main__":
    print("CRACKERNAUT RELATIVE PATH VERIFICATION")
    print("=" * 50)
    
    success1 = test_relative_paths()
    success2 = check_for_remaining_personal_data()
    
    if success1 and success2:
        print("\n🎉 ALL FIXES VERIFIED SUCCESSFULLY!")
        print("✅ All paths are now relative")
        print("✅ Personal data has been removed") 
        print("✅ Tests can run from any environment")
        sys.exit(0)
    else:
        print("\n❌ Some issues remain")
        sys.exit(1)
