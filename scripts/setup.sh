#!/usr/bin/env bash
# Crackernaut Quick Setup Script for macOS/Linux/WSL
# This script sets up the Crackernaut project with uv dependency management
# 
# This script replaces the old pip/venv setup with modern uv-based dependency management
# 
# Usage:
#   macOS/Linux: chmod +x setup.sh && ./setup.sh
#   Windows WSL: wsl chmod +x setup.sh && wsl ./setup.sh

set -e

echo "🚀 Crackernaut Quick Setup Script"
echo "================================"
echo "Using uv for modern Python dependency management"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if uv is installed
echo -e "${YELLOW}🔍 Checking for uv installation...${NC}"
if command -v uv &> /dev/null; then
    UV_VERSION=$(uv --version)
    echo -e "${GREEN}✅ uv is already installed: $UV_VERSION${NC}"
else
    echo -e "${YELLOW}📦 Installing uv...${NC}"
    if curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo -e "${GREEN}✅ uv installed successfully!${NC}"
        echo -e "${BLUE}ℹ️  You may need to restart your terminal for PATH updates.${NC}"
        echo -e "${BLUE}   Please restart and run this script again if needed.${NC}"
        # Try to source the shell config to get uv in PATH
        export PATH="$HOME/.cargo/bin:$PATH"
    else
        echo -e "${RED}❌ Failed to install uv. Please install manually:${NC}"
        echo -e "${RED}   curl -LsSf https://astral.sh/uv/install.sh | sh${NC}"
        exit 1
    fi
fi

# Clean up any old virtual environments
echo ""
echo -e "${YELLOW}🧹 Cleaning up old virtual environments...${NC}"
OLD_VENV_DIRS=(".venv" "venv" "env" ".env")
for dir in "${OLD_VENV_DIRS[@]}"; do
    if [[ -d "$dir" ]]; then
        echo -e "${YELLOW}🗑️  Removing old virtual environment: $dir${NC}"
        if rm -rf "$dir"; then
            echo -e "${GREEN}✅ Removed $dir${NC}"
        else
            echo -e "${YELLOW}⚠️  Warning: Could not remove $dir - please remove manually${NC}"
        fi
    fi
done

# Initialize project
echo ""
echo -e "${YELLOW}🔧 Setting up Crackernaut project...${NC}"

# Sync dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
if uv sync; then
    echo -e "${GREEN}✅ Basic dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install basic dependencies${NC}"
    exit 1
fi

# Install extras if requested
EXTRAS=("cuda" "dev" "test")
for extra in "${EXTRAS[@]}"; do
    echo -n "Install $extra dependencies? (y/n) [default: y]: "
    read -r response
    if [[ -z "$response" || "$response" == "y" || "$response" == "Y" ]]; then
        if uv sync --extra "$extra"; then
            echo -e "${GREEN}✅ $extra dependencies installed${NC}"
        else
            echo -e "${YELLOW}⚠️  Warning: Failed to install $extra dependencies${NC}"
        fi
    fi
done

# Test installation
echo ""
echo -e "${YELLOW}🧪 Testing installation...${NC}"
if PYTHON_VERSION=$(uv run python --version); then
    echo -e "${GREEN}✅ Python: $PYTHON_VERSION${NC}"
else
    echo -e "${RED}❌ Python test failed${NC}"
fi

if TORCH_TEST=$(uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"); then
    echo -e "${GREEN}✅ $TORCH_TEST${NC}"
else
    echo -e "${YELLOW}⚠️  Warning: PyTorch import failed${NC}"
fi

# Show completion message
echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo -e "${GREEN}=================${NC}"
echo ""
echo -e "${CYAN}📋 Next steps:${NC}"
echo -e "  • Open project in VS Code"
echo -e "  • Run: uv run python crackernaut.py --help"
echo -e "  • Use VS Code tasks via Ctrl+Shift+P > 'Tasks: Run Task'"
echo ""
echo -e "${CYAN}💡 Quick commands:${NC}"
echo -e "  • Run Crackernaut: uv run python crackernaut.py --password test123"
echo -e "  • Check GPU: uv run python check_gpu.py"
echo -e "  • Run tests: uv run python -m pytest"
echo -e "  • Format code: uv run black *.py"
echo -e "  • Lint code: uv run flake8 *.py"
echo ""
echo -e "${BLUE}📚 Learn more about uv: https://docs.astral.sh/uv/${NC}"
echo ""
echo -e "${GREEN}🔄 Migration note: Old pip/venv environments have been cleaned up.${NC}"
echo -e "${GREEN}   Run 'uv run python migrate_to_uv.py' for detailed migration if needed.${NC}"
