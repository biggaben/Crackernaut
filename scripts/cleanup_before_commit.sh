#!/bin/bash
# Clean up script for Crackernaut project
# This script removes common problematic files before committing to GitHub

echo -e "\033[36m🧹 Cleaning up Crackernaut project before commit...\033[0m"

# Get the root directory of the project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR" || exit 1

# Clean Python cache files
echo -e "\033[32mRemoving Python cache files...\033[0m"
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete

# Clean Jupyter Notebook checkpoints
echo -e "\033[32mRemoving Jupyter checkpoints...\033[0m"
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

# Clean PyTorch cache
echo -e "\033[32mClearing PyTorch cache...\033[0m"
uv run python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true

# Clean logs and temporary files
echo -e "\033[32mRemoving logs and temporary files...\033[0m"
find . -name "*.log" -type f -delete
find . -name "*.tmp" -type f -delete
find . -name "*.temp" -type f -delete
find . -name "*.bak" -type f -delete
find . -name "*.swp" -type f -delete

# Check for potential sensitive files
echo -e "\033[33mChecking for potentially sensitive files...\033[0m"
SENSITIVE_PATTERNS=("*password*" "*secret*" "*credential*" "*.key" "*.pem" "*.env")

FOUND_SENSITIVE_FILES=false
for pattern in "${SENSITIVE_PATTERNS[@]}"; do
    sensitive_files=$(find . -name "$pattern" -type f 2>/dev/null)
    if [ -n "$sensitive_files" ]; then
        FOUND_SENSITIVE_FILES=true
        echo -e "\033[31m⚠️ Warning: Found potentially sensitive files matching '$pattern':\033[0m"
        echo "$sensitive_files" | while read -r file; do
            echo -e "\033[31m  - $file\033[0m"
        done
    fi
done

if [ "$FOUND_SENSITIVE_FILES" = true ]; then
    echo -e "\033[31mPlease review these files before committing!\033[0m"
else
    echo -e "\033[32mNo potentially sensitive files found.\033[0m"
fi

# Run the Python check script for more advanced checks
echo -e "\033[36mRunning detailed file check script...\033[0m"
uv run python scripts/check_staged_files.py

echo -e "\033[32m✅ Cleanup complete!\033[0m"