#!/usr/bin/env python3
"""
Cleanup script for removing old virtual environments and updating to uv.

This script removes old virtual environment directories and ensures
the project is properly set up with uv.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str, check: bool = False) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent
        )
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            if not check:  # Only show error if we're not using check=True
                print(f"⚠️  {description} completed with warnings")
                if result.stderr.strip():
                    print(f"   Warning: {result.stderr.strip()}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"❌ Command not found for: {description}")
        return False


def remove_old_venvs() -> bool:
    """Remove old virtual environment directories."""
    print("🧹 Cleaning up old virtual environments...")
    
    # Don't include .venv as it's managed by uv
    venv_dirs = ["venv", "env", ".env", "ENV", "venv.bak", "virtualenv"]
    removed_any = False
    
    for venv_dir in venv_dirs:
        venv_path = Path(venv_dir)
        if venv_path.exists():
            print(f"🗑️  Removing old virtual environment: {venv_dir}")
            try:
                if venv_path.is_dir():
                    shutil.rmtree(venv_path)
                else:
                    venv_path.unlink()
                print(f"✅ Removed {venv_dir}")
                removed_any = True
            except OSError as e:
                print(f"❌ Failed to remove {venv_dir}: {e}")
                print(f"   Please remove {venv_dir} manually")
                return False
    
    # Note about .venv - this is managed by uv
    if Path(".venv").exists():
        print("ℹ️  Found .venv directory - this is managed by uv (leave as-is)")
    
    if not removed_any:
        print("✅ No old virtual environments found")
    
    return True


def clean_cache() -> bool:
    """Clean up Python cache files."""
    print("🧹 Cleaning up Python cache files...")
    
    cache_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo",
        "*.pyd",
        ".pytest_cache",
        ".coverage",
        "*.egg-info"
    ]
    
    removed_any = False
    
    for pattern in cache_patterns:
        if pattern == "__pycache__":
            # Find and remove __pycache__ directories
            for pycache_dir in Path(".").rglob("__pycache__"):
                try:
                    shutil.rmtree(pycache_dir)
                    print(f"✅ Removed {pycache_dir}")
                    removed_any = True
                except OSError as e:
                    print(f"⚠️  Could not remove {pycache_dir}: {e}")
        elif pattern.startswith("*."):
            # Find and remove files with specific extensions
            for file_path in Path(".").rglob(pattern):
                try:
                    file_path.unlink()
                    print(f"✅ Removed {file_path}")
                    removed_any = True
                except OSError as e:
                    print(f"⚠️  Could not remove {file_path}: {e}")
        else:
            # Find and remove directories with specific names
            for dir_path in Path(".").rglob(pattern):
                if dir_path.is_dir():
                    try:
                        shutil.rmtree(dir_path)
                        print(f"✅ Removed {dir_path}")
                        removed_any = True
                    except OSError as e:
                        print(f"⚠️  Could not remove {dir_path}: {e}")
    
    if not removed_any:
        print("✅ No cache files found to clean")
    
    return True


def check_uv_installation() -> bool:
    """Check if uv is installed and working."""
    print("🔍 Checking uv installation...")
    
    if run_command(["uv", "--version"], "Check uv version"):
        return True
    
    print("📦 uv not found. Please install uv first:")
    print("   Windows: powershell -c 'irm https://astral.sh/uv/install.ps1 | iex'")
    print("   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
    return False


def main():
    """Main cleanup function."""
    print("🧹 Crackernaut Environment Cleanup")
    print("==================================")
    print("This script will remove old virtual environments and clean up cache files.")
    print("")
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the Crackernaut project root.")
        sys.exit(1)
    
    # Remove old virtual environments
    if not remove_old_venvs():
        print("❌ Failed to remove old virtual environments")
        sys.exit(1)
    
    # Clean cache files
    if not clean_cache():
        print("⚠️  Warning: Could not clean all cache files")
    
    # Check uv installation
    if not check_uv_installation():
        print("❌ uv is not available. Please install uv and try again.")
        sys.exit(1)
    
    print("")
    print("🎉 Cleanup completed successfully!")
    print("==========================================")
    print("")
    print("📋 Next steps:")
    print("  • Run: uv sync --all-extras")
    print("  • Or use setup script: .\\setup.ps1 (Windows) or ./setup.sh (Unix)")
    print("")
    print("💡 Verify installation:")
    print("  • Check dependencies: uv run python -c \"import torch; print('PyTorch:', torch.__version__)\"")
    print("  • Check GPU: uv run python check_gpu.py")
    print("  • Run tests: uv run python -m pytest")


if __name__ == "__main__":
    main()
