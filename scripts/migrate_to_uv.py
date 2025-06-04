#!/usr/bin/env python3
"""
Legacy migration and verification script for Crackernaut project.

Note: This project has been fully migrated to uv. This script is now primarily
used for cleanup of old environments and verification of uv setup.

For new installations, simply run: uv sync
For cleanup, use the dedicated cleanup_old_envs.py script.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=True, 
            text=True,
            cwd=Path(__file__).parent
        )
        print(f"✅ {description} completed successfully")
        if result.stdout.strip():
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stderr:
            print(f"   Error: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        print(f"❌ {description} failed: Command not found")
        return False


def check_uv_installed() -> bool:
    """Check if uv is installed."""
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def install_uv() -> bool:
    """Install uv using the official installer."""
    print("📦 Installing uv...")
    
    if sys.platform.startswith("win"):
        # Windows installation
        cmd = [
            "powershell", "-ExecutionPolicy", "ByPass", "-c",
            "irm https://astral.sh/uv/install.ps1 | iex"
        ]
    else:
        # macOS/Linux installation
        cmd = "curl -LsSf https://astral.sh/uv/install.sh | sh"
    
    return run_command(cmd, "Installing uv", shell=True)


def remove_old_venv() -> bool:
    """Remove old virtual environment directories."""
    venv_dirs = [".venv", "venv", "env"]
    removed_any = False
    
    for venv_dir in venv_dirs:
        venv_path = Path(venv_dir)
        if venv_path.exists():
            print(f"🗑️  Removing old virtual environment: {venv_dir}")
            try:
                shutil.rmtree(venv_path)
                removed_any = True
                print(f"✅ Removed {venv_dir}")
            except Exception as e:
                print(f"❌ Failed to remove {venv_dir}: {e}")
                return False
    
    if not removed_any:
        print("ℹ️  No old virtual environments found to remove")
    
    return True


def backup_requirements_txt() -> bool:
    """Backup the old requirements.txt file."""
    req_file = Path("requirements.txt")
    if req_file.exists():
        backup_file = Path("requirements.txt.backup")
        try:
            shutil.copy2(req_file, backup_file)
            print(f"✅ Backed up requirements.txt to {backup_file}")
            return True
        except Exception as e:
            print(f"❌ Failed to backup requirements.txt: {e}")
            return False
    else:
        print("ℹ️  No requirements.txt found to backup")
        return True


def init_uv_project() -> bool:
    """Initialize uv project."""
    if not run_command(["uv", "sync"], "Initializing uv project and syncing dependencies"):
        return False
    
    # Also sync with extras for complete setup
    extras = ["cuda", "dev", "test"]
    for extra in extras:
        run_command(
            ["uv", "sync", "--extra", extra], 
            f"Syncing {extra} dependencies"
        )
    
    return True


def verify_installation() -> bool:
    """Verify the installation works."""
    print("🔍 Verifying installation...")
    
    # Check Python version
    if not run_command(["uv", "run", "python", "--version"], "Checking Python version"):
        return False
    
    # Check core dependencies
    test_imports = [
        "import torch; print(f'PyTorch: {torch.__version__}')",
        "import numpy; print(f'NumPy: {numpy.__version__}')",
        "import sklearn; print(f'scikit-learn: {sklearn.__version__}')",
    ]
    
    for test_import in test_imports:
        if not run_command(
            ["uv", "run", "python", "-c", test_import], 
            f"Testing import: {test_import.split(';')[0]}"
        ):
            print(f"⚠️  Warning: Failed to import {test_import.split(';')[0]}")
    
    return True


def show_migration_summary():
    """Show verification summary and usage information."""
    print("\n" + "="*60)
    print("🎉 MIGRATION VERIFICATION COMPLETED!")
    print("="*60)
    print()
    print("ℹ️  Project Status:")
    print("  • ✅ Project is already using uv for dependency management")
    print("  • ✅ Dependencies are managed via pyproject.toml") 
    print("  • ✅ Virtual environment is automatically managed by uv")
    print("  • ℹ️  requirements.txt is maintained for legacy/CI compatibility")
    print()
    print("🚀 Common commands:")
    print("  • Install dependencies: uv sync")
    print("  • Run Python scripts: uv run python <script>")
    print("  • Add new dependencies: uv add <package>")
    print("  • Remove dependencies: uv remove <package>")
    print("  • Install with extras: uv sync --extra cuda")
    print()  
    print("🧹 Cleanup tools:")
    print("  • Clean old environments: python cleanup_old_envs.py")
    print("  • Clean Python cache: Remove-Item -Recurse __pycache__, *.pyc")
    print()
    print("📚 Learn more about uv: https://docs.astral.sh/uv/")


def main():
    """Main migration/verification function."""
    print("🔄 Crackernaut Legacy Migration & Verification Tool")
    print("=" * 50)
    print()
    print("ℹ️  Note: This project is already migrated to uv!")
    print("   For new setups, just run: uv sync")
    print("   For cleanup, use: python cleanup_old_envs.py")
    print()
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the Crackernaut project root.")
        sys.exit(1)
      # Step 1: Check if uv is installed
    if not check_uv_installed():
        print("📦 uv is not installed. Installing...")
        if not install_uv():
            print("❌ Failed to install uv. Please install manually and try again.")
            print("   Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"")
            print("   Windows (WSL): wsl curl -LsSf https://astral.sh/uv/install.sh | sh")
            print("   macOS/Linux: curl -LsSf https://astral.sh/uv/install.sh | sh")
            sys.exit(1)
        
        # After installation, we might need to restart or update PATH
        print("✅ uv installed! You may need to restart your terminal or VS Code.")
        print("   Please restart and run this script again.")
        sys.exit(0)
    else:
        print("✅ uv is already installed")
    
    # Step 2: Backup old requirements.txt
    if not backup_requirements_txt():
        print("❌ Failed to backup requirements.txt")
        sys.exit(1)
    
    # Step 3: Remove old virtual environments
    if not remove_old_venv():
        print("❌ Failed to clean up old virtual environments")
        sys.exit(1)
    
    # Step 4: Initialize uv project
    if not init_uv_project():
        print("❌ Failed to initialize uv project")
        sys.exit(1)
    
    # Step 5: Verify installation
    if not verify_installation():
        print("⚠️  Installation verification had some issues, but migration may still be successful")
    
    # Step 6: Show summary
    show_migration_summary()


if __name__ == "__main__":
    main()
