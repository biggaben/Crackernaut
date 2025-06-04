# Development Utilities

This directory contains development and debugging utilities for Crackernaut.

## Available Scripts

### CUDA/PyTorch Testing

- **`check_cuda.py`** - Basic CUDA availability check
- **`debug_torch.py`** - Detailed PyTorch debugging information
- **`simple_cuda_test.py`** - Simple CUDA functionality test
- **`simple_torch_test.py`** - Basic PyTorch installation test

## Usage

Run these scripts from the project root:

```bash
# Check CUDA availability
uv run python scripts/dev/check_cuda.py

# Debug PyTorch installation
uv run python scripts/dev/debug_torch.py

# Test CUDA functionality
uv run python scripts/dev/simple_cuda_test.py

# Test PyTorch basics
uv run python scripts/dev/simple_torch_test.py
```

## When to Use

- **Setting up development environment** - Run all scripts to verify setup
- **Debugging GPU issues** - Use CUDA-specific scripts
- **CI/CD pipeline troubleshooting** - Validate PyTorch installation
- **Performance investigation** - Check GPU availability and configuration

## Integration with Main Scripts

For production use, prefer the main project scripts:

```bash
# Production GPU check (with more comprehensive output)
uv run python -m pytest tests/test_torch.py -v

# Or use the VS Code task
# Command Palette > "Tasks: Run Task" > "Check GPU Status"
```
