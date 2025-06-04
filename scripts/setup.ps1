#!/usr/bin/env pwsh
# Crackernaut Quick Setup Script for Windows (PowerShell)
# This script sets up the Crackernaut project with uv dependency management
# 
# This script replaces the old pip/venv setup with modern uv-based dependency management

Write-Host "🚀 Crackernaut Quick Setup Script" -ForegroundColor Cyan
Write-Host "================================" -ForegroundColor Cyan
Write-Host "Using uv for modern Python dependency management" -ForegroundColor Blue
Write-Host ""

# Check if uv is installed
Write-Host "🔍 Checking for uv installation..." -ForegroundColor Yellow
try {
    $uvVersion = uv --version
    Write-Host "✅ uv is already installed: $uvVersion" -ForegroundColor Green
} catch {
    Write-Host "📦 Installing uv..." -ForegroundColor Yellow
    try {
        irm https://astral.sh/uv/install.ps1 | iex
        Write-Host "✅ uv installed successfully!" -ForegroundColor Green
        Write-Host "ℹ️  You may need to restart your terminal for PATH updates." -ForegroundColor Blue
        Write-Host "   Please restart and run this script again if needed." -ForegroundColor Blue
    } catch {
        Write-Host "❌ Failed to install uv. Please install manually:" -ForegroundColor Red
        Write-Host "   powershell -c 'irm https://astral.sh/uv/install.ps1 | iex'" -ForegroundColor Red
        exit 1
    }
}

# Clean up any old virtual environments
Write-Host ""
Write-Host "🧹 Cleaning up old virtual environments..." -ForegroundColor Yellow
$oldVenvDirs = @(".venv", "venv", "env", ".env")
foreach ($dir in $oldVenvDirs) {
    if (Test-Path $dir) {
        try {
            Remove-Item -Recurse -Force $dir
            Write-Host "✅ Removed old virtual environment: $dir" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Warning: Could not remove $dir - please remove manually" -ForegroundColor Yellow
        }
    }
}

# Initialize project
Write-Host ""
Write-Host "🔧 Setting up Crackernaut project..." -ForegroundColor Yellow

# Sync dependencies
Write-Host "📦 Installing dependencies..." -ForegroundColor Yellow
try {
    uv sync
    Write-Host "✅ Basic dependencies installed" -ForegroundColor Green
} catch {
    Write-Host "❌ Failed to install basic dependencies" -ForegroundColor Red
    exit 1
}

# Install extras if requested
$extras = @("cuda", "dev", "test")
foreach ($extra in $extras) {
    $response = Read-Host "Install $extra dependencies? (y/n) [default: y]"
    if ($response -eq "" -or $response -eq "y" -or $response -eq "Y") {
        try {
            uv sync --extra $extra
            Write-Host "✅ $extra dependencies installed" -ForegroundColor Green
        } catch {
            Write-Host "⚠️  Warning: Failed to install $extra dependencies" -ForegroundColor Yellow
        }
    }
}

# Test installation
Write-Host ""
Write-Host "🧪 Testing installation..." -ForegroundColor Yellow
try {
    $pythonVersion = uv run python --version
    Write-Host "✅ Python: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python test failed" -ForegroundColor Red
}

try {
    $torchTest = uv run python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    Write-Host "✅ $torchTest" -ForegroundColor Green
} catch {
    Write-Host "⚠️  Warning: PyTorch import failed" -ForegroundColor Yellow
}

# Show completion message
Write-Host ""
Write-Host "🎉 Setup Complete!" -ForegroundColor Green
Write-Host "=================" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Cyan
Write-Host "  • Open project in VS Code" -ForegroundColor White
Write-Host "  • Run: uv run python crackernaut.py --help" -ForegroundColor White
Write-Host "  • Use VS Code tasks via Ctrl+Shift+P > 'Tasks: Run Task'" -ForegroundColor White
Write-Host ""
Write-Host "💡 Quick commands:" -ForegroundColor Cyan
Write-Host "  • Run Crackernaut: uv run python crackernaut.py --password test123" -ForegroundColor White
Write-Host "  • Check GPU: uv run python check_gpu.py" -ForegroundColor White
Write-Host "  • Run tests: uv run python -m pytest" -ForegroundColor White
Write-Host "  • Format code: uv run black *.py" -ForegroundColor White
Write-Host "  • Lint code: uv run flake8 *.py" -ForegroundColor White
Write-Host ""
Write-Host "📚 Learn more about uv: https://docs.astral.sh/uv/" -ForegroundColor Blue
Write-Host ""
Write-Host "🔄 Migration note: Old pip/venv environments have been cleaned up." -ForegroundColor Green
Write-Host "   Run 'uv run python migrate_to_uv.py' for detailed migration if needed." -ForegroundColor Green
