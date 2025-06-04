# Project Structure

## Overview

Crackernaut has been restructured for better organization and maintainability. Below is the new directory structure:

```text
Crackernaut/
├── src/                          # Main source code
│   ├── __init__.py              # Package initialization
│   ├── cuda_ml.py               # CUDA ML operations
│   ├── distributed_training.py  # Distributed training support
│   ├── list_preparer.py         # Data preparation utilities
│   ├── models/                  # ML model implementations
│   │   ├── __init__.py
│   │   ├── embedding/           # Password embedding models
│   │   │   └── embedding_model.py
│   │   └── transformer/         # Transformer models
│   │       ├── transformer_model.py
│   │       └── transformer_model.pth
│   └── utils/                   # Utility modules
│       ├── __init__.py
│       ├── async_utils.py       # Async I/O utilities
│       ├── common_utils.py      # Common utility functions
│       ├── config_utils.py      # Configuration management
│       ├── performance_utils.py # Performance monitoring
│       └── variant_utils.py     # Password variant generation
├── scripts/                     # Setup and utility scripts
│   ├── check_gpu.py            # GPU status checker
│   ├── cleanup_old_envs.py     # Environment cleanup
│   ├── migrate_to_uv.py        # Legacy migration script
│   ├── setup.ps1               # Windows setup script
│   └── setup.sh                # Unix setup script
├── trainingdata/               # Training datasets (not in VCS)
├── clusters/                   # Processed clustering data
├── .vscode/                    # VS Code configuration
├── .github/                    # GitHub configuration
├── crackernaut.py             # Main application entry point
├── crackernaut_train.py       # Training script
├── test_variants.py           # Test suite
├── config.json               # Application configuration
├── pyproject.toml           # Project dependencies (uv)
├── requirements.txt         # Legacy dependencies (deprecated)
└── README.md               # Project documentation
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
