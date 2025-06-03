#!/usr/bin/env python
"""
check_staged_files.py - Identify problematic files before committing to GitHub

This script identifies files that shouldn't be committed to GitHub, including:
- Large files (configurable threshold)
- Binary files that might contain sensitive data
- Files matching patterns that could contain passwords or credentials
- Cache files and temporary data

Usage:
    uv run python scripts/check_staged_files.py [--max-size SIZE_MB]

Example:
    uv run python scripts/check_staged_files.py --max-size 10
"""

import os
import sys
import re
import argparse
from pathlib import Path
import mimetypes
from typing import List, Dict, Set, Tuple, Optional


class GitIgnorePatternMatcher:
    """Match files against gitignore patterns."""
    
    def __init__(self, gitignore_path: str):
        self.patterns = self._load_patterns(gitignore_path)
    
    def _load_patterns(self, gitignore_path: str) -> List[str]:
        """Load patterns from gitignore file."""
        if not os.path.exists(gitignore_path):
            return []
        
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process patterns, removing comments and empty lines
        patterns = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#'):
                # Handle negated patterns with !
                patterns.append(line)
                
        return patterns
    
    def is_ignored(self, file_path: str) -> bool:
        """Check if a file matches any gitignore pattern."""
        rel_path = os.path.normpath(file_path)
        
        for pattern in self.patterns:
            negated = pattern.startswith('!')
            if negated:
                pattern = pattern[1:]
            
            # Convert gitignore glob pattern to regex
            regex_pattern = self._glob_to_regex(pattern)
            matches = re.search(regex_pattern, rel_path) is not None
            
            if matches and not negated:
                return True
            if matches and negated:
                return False
                
        return False
    
    def _glob_to_regex(self, pattern: str) -> str:
        """Convert gitignore glob pattern to regex."""
        # Escape special regex chars
        pattern = re.escape(pattern)
        
        # Convert gitignore glob patterns to regex
        pattern = pattern.replace(r'\*\*', '.*')  # ** matches anything
        pattern = pattern.replace(r'\*', '[^/]*')  # * matches anything except /
        pattern = pattern.replace(r'\?', '[^/]')   # ? matches single character except /
        
        # Handle directory-only patterns ending with /
        if pattern.endswith(r'\/'):
            pattern = pattern[:-2] + '(?:/.*)?$'
        else:
            pattern = pattern + '(?:$|/.*)'
            
        return f'^{pattern}'


def is_binary_file(file_path: str) -> bool:
    """Check if a file is binary."""
    mime, _ = mimetypes.guess_type(file_path)
    if mime is None:
        # If we can't determine the type, check the content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Try to read as text
                return False
        except UnicodeDecodeError:
            return True
    
    return mime is not None and not mime.startswith(('text/', 'application/json'))


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def find_problematic_files(
    root_dir: str, 
    max_size_mb: float = 5.0
) -> Dict[str, List[str]]:
    """Find files that should not be committed to GitHub."""
    
    gitignore_matcher = GitIgnorePatternMatcher(os.path.join(root_dir, '.gitignore'))
    
    # Define problematic patterns (files that might contain sensitive data)
    sensitive_patterns = [
        r'\.env$', 
        r'.*_key\..*', 
        r'.*password.*', 
        r'.*secret.*',
        r'.*credential.*',
        r'.*token.*\.txt$',
        r'.*\.pem$',
        r'.*\.key$'
    ]
    
    # Directories to skip entirely
    skip_dirs = {'.git', '.github', '__pycache__', '.pytest_cache', '.vscode', '.idea'}
    
    results = {
        'large_files': [],
        'binary_files': [],
        'sensitive_files': [],
        'untracked_in_gitignore': []
    }
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip directories we want to exclude
        dirnames[:] = [d for d in dirnames if d not in skip_dirs]
        
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(file_path, root_dir)
            
            # Skip files already properly handled by gitignore
            if gitignore_matcher.is_ignored(rel_path):
                continue
                
            # Check file size
            if os.path.exists(file_path):
                file_size_mb = get_file_size_mb(file_path)
                if file_size_mb > max_size_mb:
                    results['large_files'].append(f"{rel_path} ({file_size_mb:.2f} MB)")
                
                # Check if it's a binary file
                if is_binary_file(file_path):
                    results['binary_files'].append(rel_path)
                
                # Check for sensitive patterns
                for pattern in sensitive_patterns:
                    if re.search(pattern, rel_path, re.IGNORECASE):
                        results['sensitive_files'].append(rel_path)
                        break
                        
                # Files that should likely be in gitignore
                if any(ext in filename for ext in ['.log', '.tmp', '.temp', '.swp', '.bak']):
                    results['untracked_in_gitignore'].append(rel_path)
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Check for problematic files before committing')
    parser.add_argument('--max-size', type=float, default=5.0,
                      help='Maximum file size in MB (default: 5.0)')
    args = parser.parse_args()
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results = find_problematic_files(root_dir, args.max_size)
    
    # Check if we found any problematic files
    has_issues = any(files for files in results.values())
    
    if has_issues:
        print("\n==== 🔍 Potential Issues Found ====\n")
        
        if results['large_files']:
            print("🚨 LARGE FILES (might not belong in git):")
            for file in results['large_files']:
                print(f"  - {file}")
            print()
        
        if results['binary_files']:
            print("⚠️ BINARY FILES (check if these should be committed):")
            for file in results['binary_files']:
                print(f"  - {file}")
            print()
            
        if results['sensitive_files']:
            print("🔐 POTENTIALLY SENSITIVE FILES:")
            for file in results['sensitive_files']:
                print(f"  - {file}")
            print()
            
        if results['untracked_in_gitignore']:
            print("📝 FILES THAT SHOULD PROBABLY BE IN .gitignore:")
            for file in results['untracked_in_gitignore']:
                print(f"  - {file}")
            print()
            
        print("""
Recommendation:
1. Review these files and remove any that shouldn't be committed
2. Update .gitignore to exclude problematic patterns
3. For large files, consider git-lfs or alternative storage

For security research projects like Crackernaut, be especially careful with:
- Password data files and training datasets
- Credential files or tokens
- Large binary files that might contain sensitive information
""")
        return 1
    else:
        print("✅ No problematic files found. Safe to commit!")
        return 0


if __name__ == "__main__":
    sys.exit(main())