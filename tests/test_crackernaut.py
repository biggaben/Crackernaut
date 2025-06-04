#!/usr/bin/env python3
"""Simple wrapper to test crackernaut.py execution."""

import subprocess
import sys

try:
    result = subprocess.run([
        sys.executable, "crackernaut.py", "--help"
    ], capture_output=True, text=True, cwd=".")
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    print(f"Return code: {result.returncode}")
    
except Exception as e:
    print(f"Error running command: {e}")
