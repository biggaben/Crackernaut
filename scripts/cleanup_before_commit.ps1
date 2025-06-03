# Clean up script for Crackernaut project
# This script removes common problematic files before committing to GitHub

Write-Host "🧹 Cleaning up Crackernaut project before commit..." -ForegroundColor Cyan

# Get the root directory of the project
$RootDir = Split-Path -Parent (Split-Path -Parent $PSCommandPath)
Push-Location $RootDir

# Clean Python cache files
Write-Host "Removing Python cache files..." -ForegroundColor Green
Remove-Item -Recurse -Force -ErrorAction SilentlyContinue -Path "__pycache__"
Get-ChildItem -Path . -Filter "__pycache__" -Recurse -Directory | Remove-Item -Recurse -Force
Get-ChildItem -Path . -Filter "*.pyc" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.pyo" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.pyd" -Recurse -File | Remove-Item -Force

# Clean Jupyter Notebook checkpoints
Write-Host "Removing Jupyter checkpoints..." -ForegroundColor Green
Get-ChildItem -Path . -Filter ".ipynb_checkpoints" -Recurse -Directory | Remove-Item -Recurse -Force

# Clean PyTorch cache
Write-Host "Clearing PyTorch cache..." -ForegroundColor Green
uv run python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" -ErrorAction SilentlyContinue

# Clean logs and temporary files
Write-Host "Removing logs and temporary files..." -ForegroundColor Green
Get-ChildItem -Path . -Filter "*.log" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.tmp" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.temp" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.bak" -Recurse -File | Remove-Item -Force
Get-ChildItem -Path . -Filter "*.swp" -Recurse -File | Remove-Item -Force

# Check for potential sensitive files
Write-Host "Checking for potentially sensitive files..." -ForegroundColor Yellow
$sensitivePatterns = @(
    "*password*",
    "*secret*",
    "*credential*",
    "*.key",
    "*.pem",
    "*.env"
)

$foundSensitiveFiles = $false
foreach ($pattern in $sensitivePatterns) {
    $sensitiveFiles = Get-ChildItem -Path . -Filter $pattern -Recurse -File -ErrorAction SilentlyContinue
    if ($sensitiveFiles) {
        $foundSensitiveFiles = $true
        Write-Host "⚠️ Warning: Found potentially sensitive files matching '$pattern':" -ForegroundColor Red
        $sensitiveFiles | ForEach-Object {
            Write-Host "  - $($_.FullName)" -ForegroundColor Red
        }
    }
}

if ($foundSensitiveFiles) {
    Write-Host "Please review these files before committing!" -ForegroundColor Red
} else {
    Write-Host "No potentially sensitive files found." -ForegroundColor Green
}

# Run the Python check script for more advanced checks
Write-Host "Running detailed file check script..." -ForegroundColor Cyan
uv run python scripts/check_staged_files.py

Pop-Location
Write-Host "✅ Cleanup complete!" -ForegroundColor Green