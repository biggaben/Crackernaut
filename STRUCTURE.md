# Project Structure

## Overview

Crackernaut has been restructured for better organization and maintainability. The project now includes comprehensive PyTorch CUDA support and modern dependency management with uv.

## CUDA and GPU Configuration

The project is pre-configured for high-performance GPU acceleration:

- **PyTorch CUDA 12.1**: Optimized for NVIDIA RTX 3090 and similar GPUs
- **Automatic CUDA Detection**: Falls back to CPU if GPU unavailable
- **NVIDIA Runtime Libraries**: All required CUDA libraries automatically installed
- **GPU Memory Management**: Efficient memory usage patterns for large datasets

### CUDA Package Configuration

```toml
[project.optional-dependencies]
cuda = [
    "torch==2.5.1+cu121",
    "torchvision==0.20.1+cu121", 
    "torchaudio==2.5.1+cu121",
]

[tool.uv.sources]
torch = { index = "pytorch-cuda" }
torchvision = { index = "pytorch-cuda" }
torchaudio = { index = "pytorch-cuda" }

[[tool.uv.index]]
name = "pytorch-cuda"
url = "https://download.pytorch.org/whl/cu121"
```

## Changes Made

### Removed Files

- `models/mlp/mlp_model.py` - Empty file
- `models/rnn/rnn_model.py` - Empty file
- `models/rnn/rnn_model.pth` - Orphaned model weights
- `models/mlp/` and `models/rnn/` directories - Empty implementations

### Reorganized Files

- All utility modules moved to `src/utils/`
- Model implementations moved to `src/models/`
- Core application modules moved to `src/`
- Setup and utility scripts moved to `scripts/`

### Updated References

- All import statements updated to reflect new structure
- VS Code tasks updated for new file locations
- Documentation updated with new paths
- README.md updated for new setup script locations

## Import Changes

### Before

```python
from config_utils import load_configuration
from variant_utils import generate_variants
from models.transformer.transformer_model import PasswordTransformer
```

### After

```python
from src.utils.config_utils import load_configuration
from src.utils.variant_utils import generate_variants
from src.models.transformer.transformer_model import PasswordTransformer
```

## Benefits

1. **Better Organization**: Clear separation of concerns with dedicated directories
2. **Maintainability**: Easier to locate and maintain specific functionality
3. **Scalability**: Room for growth without cluttering the root directory
4. **Professional Structure**: Follows Python packaging best practices
5. **Reduced Clutter**: Scripts and utilities organized separately from core code

## Development Workflow

All existing VS Code tasks and development workflows remain the same, but now reference the updated file locations.
