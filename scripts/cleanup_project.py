#!/usr/bin/env python3
"""
Crackernaut Project Cleanup Script

This script performs a comprehensive cleanup of the project directory,
removing cache files, temporary directories, and other artifacts.
"""

import shutil
from pathlib import Path
from typing import List


def remove_cache_directories(root_path: Path) -> List[str]:
    """Remove Python cache directories recursively."""
    removed = []
    
    # Find and remove __pycache__ directories
    for pycache_dir in root_path.rglob("__pycache__"):
        if pycache_dir.is_dir():
            shutil.rmtree(pycache_dir)
            removed.append(str(pycache_dir.relative_to(root_path)))
    
    return removed


def remove_python_cache_files(root_path: Path) -> List[str]:
    """Remove individual Python cache files."""
    removed = []
    extensions = [".pyc", ".pyo", ".pyd"]
    
    for ext in extensions:
        for cache_file in root_path.rglob(f"*{ext}"):
            if cache_file.is_file():
                cache_file.unlink()
                removed.append(str(cache_file.relative_to(root_path)))
    
    return removed


def remove_other_cache_directories(root_path: Path) -> List[str]:
    """Remove other cache and temporary directories."""
    removed = []
    cache_dirs = [
        ".mypy_cache",
        ".pytest_cache", 
        ".coverage",
        "htmlcov",
        ".tox",
        "build",
        "dist",
        ".DS_Store",
        "Thumbs.db"
    ]
    
    for cache_dir_name in cache_dirs:
        for cache_dir in root_path.rglob(cache_dir_name):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)
                removed.append(str(cache_dir.relative_to(root_path)))
            elif cache_dir.is_file():
                cache_dir.unlink()
                removed.append(str(cache_dir.relative_to(root_path)))
    
    return removed


def remove_temporary_files(root_path: Path) -> List[str]:
    """Remove temporary files."""
    removed = []
    temp_patterns = ["*.tmp", "*.temp", "*.swp", "*.swo", "*~"]
    
    for pattern in temp_patterns:
        for temp_file in root_path.rglob(pattern):
            if temp_file.is_file():
                temp_file.unlink()
                removed.append(str(temp_file.relative_to(root_path)))
    
    return removed


def main():
    """Main cleanup function."""
    project_root = Path(__file__).parent.parent
    print(f"Cleaning up project directory: {project_root}")
    print("=" * 60)
    
    # Remove Python cache directories
    print("Removing Python cache directories...")
    removed_cache_dirs = remove_cache_directories(project_root)
    for item in removed_cache_dirs:
        print(f"  Removed: {item}")
    
    # Remove Python cache files
    print("\nRemoving Python cache files...")
    removed_cache_files = remove_python_cache_files(project_root)
    for item in removed_cache_files:
        print(f"  Removed: {item}")
    
    # Remove other cache directories
    print("\nRemoving other cache directories...")
    removed_other_cache = remove_other_cache_directories(project_root)
    for item in removed_other_cache:
        print(f"  Removed: {item}")
    
    # Remove temporary files
    print("\nRemoving temporary files...")
    removed_temp = remove_temporary_files(project_root)
    for item in removed_temp:
        print(f"  Removed: {item}")
    
    # Summary
    total_removed = (
        len(removed_cache_dirs) + 
        len(removed_cache_files) + 
        len(removed_other_cache) + 
        len(removed_temp)
    )
    
    print("=" * 60)
    print(f"Cleanup complete! Removed {total_removed} items.")
    
    if total_removed == 0:
        print("Project directory was already clean.")
    else:
        print("Project directory is now clean and ready for development.")


if __name__ == "__main__":
    main()
