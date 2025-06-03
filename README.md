# Crackernaut

A sophisticated password variant generation and ML security research tool that combines rule-based transformations with machine learning to produce human-like password variants for ethical penetration testing and security analysis.

## Overview

Crackernaut is a password security research platform designed for authorized security professionals and researchers. It leverages advanced machine learning approaches including transformers, RNNs, and MLPs to analyze, score, and generate password variants that mimic human behavior patterns.

## Recent Updates

- **🎯 Major Project Restructuring (June 2025):**  
  Complete reorganization with professional directory structure (`src/`, `scripts/`), removal of unused components, and improved maintainability.
- **🧹 Comprehensive Cleanup:**  
  Removed redundant empty model files, moved utilities to organized directories, and updated all imports and references.
- **📁 Modern Project Structure:**  
  Implemented `src/utils/`, `src/models/` organization with `scripts/` directory for better code organization.
- **🔧 Enhanced uv Integration:**  
  Fully migrated to uv package manager with automated setup scripts and legacy environment cleanup utilities.
- **🤖 Advanced Transformer Architecture:**  
  Lightweight transformer-based models for richer password embeddings and more accurate variant scoring.
- **⚡ Optimized GPU Pipeline:**  
  Enhanced pipeline supporting CUDA acceleration, distributed training, and efficient GPU memory management.
- **📊 Intelligent Data Processing:**  
  Asynchronous processing of massive password datasets with Mini-Batch K-Means clustering for optimized training.
- **🚀 High-Performance Batch Processing:**  
  Memory-efficient batch processing with producer-consumer patterns for large-scale operations.

## Project Structure

Crackernaut follows a modern, organized project structure for better maintainability and development experience:

```text
Crackernaut/
├── src/                          # Main source code directory
│   ├── utils/                    # Utility modules
│   │   ├── config_utils.py       # Configuration management
│   │   ├── variant_utils.py      # Password variant generation logic
│   │   ├── async_utils.py        # Asynchronous processing utilities
│   │   ├── common_utils.py       # Common helper functions
│   │   └── performance_utils.py  # Performance monitoring utilities
│   ├── models/                   # ML model implementations
│   │   ├── embedding/            # Password embedding models
│   │   └── transformer/          # Transformer architecture models
│   ├── cuda_ml.py                # CUDA GPU acceleration utilities
│   ├── distributed_training.py   # Multi-GPU distributed training
│   └── list_preparer.py          # Password list processing and clustering
├── scripts/                      # Setup and utility scripts
│   ├── setup.ps1                 # Windows PowerShell setup script
│   ├── setup.sh                  # Unix/Linux setup script
│   ├── check_gpu.py              # GPU status verification utility
│   ├── cleanup_old_envs.py       # Legacy environment cleanup tool
│   └── migrate_to_uv.py          # Migration utilities for legacy setups
├── crackernaut.py                # Main application entry point
├── crackernaut_train.py          # ML model training pipeline
├── test_variants.py              # Password variant testing and validation
├── config.json                   # Configuration file for models and processing
├── pyproject.toml                # uv dependency management and project metadata
├── trainingdata/                 # Password datasets (excluded from git)
├── clusters/                     # Processed clustering data
├── .vscode/                      # VS Code configuration and tasks
└── .github/                      # GitHub configuration and Copilot instructions
```

### Key Organizational Benefits

- **src/**: Clean separation of main source code from scripts and configuration
- **src/utils/**: Centralized utility functions for better maintainability and reusability
- **src/models/**: Organized ML model implementations with clear architecture separation
- **scripts/**: Setup and maintenance scripts separate from application logic
- **Removed redundancy**: Eliminated empty model files and unused dependencies
- **Professional structure**: Follows Python packaging best practices for research projects
├── pyproject.toml                # Project dependencies (uv)
└── trainingdata/                 # Training datasets (not in VCS)
```

This structure provides:

- **Clear separation** of core code, utilities, and scripts
- **Professional organization** following Python packaging best practices
- **Maintainability** with logical grouping of related functionality
- **Scalability** with room for growth without cluttering the root directory

For detailed information about the structure and recent changes, see [STRUCTURE.md](STRUCTURE.md).

## Purpose

Crackernaut is a sophisticated password guessing utility designed to generate human-like password variants from a given base password. It combines rule-based transformations with machine learning to produce plausible password guesses that reflect common patterns humans use when creating passwords. This tool is intended for security researchers, penetration testers, and anyone needing to test password strength by generating realistic variants for analysis or cracking attempts.

## Features

- **Human-Like Variants:**  
  Generates passwords using transformations like numeric increments, symbol additions, capitalization changes, leet speak, shifts, repetitions, and middle insertions.
- **Machine Learning Scoring:**  
  Uses PyTorch-based models (including a **new transformer model** and legacy models such as MLP, RNN, BiLSTM) to score variants based on their likelihood of human use.
- **GPU Acceleration:**  
  Leverages CUDA for faster computation on compatible hardware with automatic device detection.
- **Smart List Preparation:**  
  Processes large password datasets using transformer-based embeddings and clustering to create optimized, diverse training sets.
- **Configurable:**  
  Adjust transformation weights, chain depth, and maximum length via a JSON configuration file.
- **Multi-Processing:**  
  Employs parallel processing for variant generation and a producer-consumer pattern for efficient processing.
- **Training Modes:**  
  - Bulk training from wordlists with automated learning.
  - Self-supervised learning from password pattern mining.
  - Intelligent dataset preparation through clustering.
- **Hyperparameter Tuning:**  
  Optimizes ML models using Bayesian optimization with the Ax library.
- **Distributed Training:**  
  Supports distributed data parallelism across multiple GPUs for faster training.
- **Asynchronous I/O:**  
  Uses asynchronous file operations for efficient data handling.
- **Flexible Output:**  
  Outputs variants to the console or a file, with options to limit quantity and length.
- **Multiple Model Options:**  
  - **NEW:** Transformer model (more accurate, supports batch processing).
  - Legacy models: MLP, RNN, BiLSTM.

## Requirements

- Python 3.8.1+
- PyTorch (automatically installed)
- CUDA (optional, for GPU acceleration)
- **uv** (modern Python package manager)

All dependencies are managed via `pyproject.toml` with optional extras:
- `cuda`: For CUDA/GPU acceleration (includes pycuda)
- `dev`: Development tools (black, flake8, mypy, pre-commit)
- `test`: Testing framework (pytest, coverage)

**Note:** The legacy `requirements.txt` file is maintained for backwards compatibility with CI/CD systems that haven't migrated to uv yet, but all new development should use uv.

### Hardware

- **Recommended:** CUDA-capable GPU (e.g., Nvidia RTX 3090) for optimal performance.
- **Minimum:** Any CPU (runs without GPU support, though slower).

## Installation

1. **Install uv** (if not already installed):
   ```bash
   # On Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone this repository:**
   ```bash
   git clone <repository-url>
   cd crackernaut
   ```

3. **Quick Setup** (Recommended):
   ```bash
   # Windows
   .\scripts\setup.ps1
   
   # macOS/Linux/WSL
   chmod +x scripts/setup.sh && ./scripts/setup.sh
   ```

4. **Manual Installation:**
   ```bash
   # Basic installation
   uv sync
   
   # With CUDA support (recommended for GPU acceleration)
   uv sync --extra cuda
   
   # With development tools
   uv sync --extra dev
   
   # All extras
   uv sync --all-extras
   ```

   **Note:** PyTorch with CUDA support will be automatically installed when using the `cuda` extra. If you need a specific PyTorch version or CUDA configuration, you can override it in your local environment.

### Migrating from Legacy pip/venv Setup

If you're upgrading from an older version that used pip and virtual environments, use the automated migration:

```bash
# Automated migration (recommended)
uv run python migrate_to_uv.py

# Or use the quick setup scripts (they clean up old environments automatically)
# Windows:
.\scripts\setup.ps1

# macOS/Linux/WSL:
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

The migration and setup scripts will:
- Remove old virtual environment directories (`.venv`, `venv`, `env`)
- Install uv if not already present
- Set up all dependencies using uv
- Update your development workflow to use uv commands

## Usage

### Basic Usage
Run Crackernaut to generate variants from a base password:
```bash
uv run python crackernaut.py --password "mypassword" --model transformer
```

### Command Line Options

- `--password, -p`: Base password to analyze.
- `--config, -c`: Path to configuration file (default: config.json).
- `--depth, -d`: Chain depth for variant generation.
- `--model, -m`: Model type: transformer, rnn, bilstm, mlp (default: transformer).
- `--prepare`: Trigger list preparation.
- `--lp-dataset`: Path to large password dataset for list preparation.
- `--lp-output`: Output directory for clusters (default: clusters).
- `--lp-chunk-size`: Chunk size for list preparation (default: 1000000).

### Training the Model
Use crackernaut_train.py to train the ML model and refine configuration.

#### Bulk Training
Train on a wordlist (one password per line):
```bash
uv run python crackernaut_train.py --wordlist <wordlist_file> [-t <iterations>] [--model <model_type>]
```

Example:
```bash
uv run python crackernaut_train.py --wordlist rockyou.txt --times 5 --model bilstm
```

#### Intelligent Dataset Preparation
Process large wordlists into optimized training sets:
```bash
uv run python crackernaut_train.py --prepare --lp-dataset <path_to_dataset> [--clusters <num_clusters>] [--lp-chunk-size <size>] [--lp-output <output_dir>]
```

Example:
```bash
uv run python crackernaut_train.py --prepare --lp-dataset breach_compilation.txt --clusters 20000 --lp-chunk-size 2000000
```

#### Interactive Training
Fine-tune the model with interactive feedback:
```bash
uv run python crackernaut_train.py --interactive
```

### Configuration
Customize Crackernaut via the config.json file. Key options include:
- `model_type`: Model for scoring (transformer, rnn, bilstm, mlp)
- `model_embed_dim`: Embedding dimension for the transformer (default: 64)
- `model_num_heads`: Number of attention heads (default: 4)
- `model_num_layers`: Number of transformer layers (default: 3)
- `model_hidden_dim`: Hidden dimension in transformer feed-forward layers (default: 128)
- `model_dropout`: Dropout rate (default: 0.2)
- `chain_depth`: Maximum number of modifications (default: 2)
- `max_length`: Maximum length for generated passwords
- `transformation_weights`: Weights for different transformation types
- `current_base`: Base password for interactive training
- `learning_rate`: Model training learning rate

### Ethical and Security Considerations
- Always obtain explicit permission before using Crackernaut for security testing.
- Handle password datasets securely, using encryption where necessary, and comply with all applicable data protection laws.

## Technical Architecture

### Modern Project Organization

Crackernaut now features a professionally structured codebase:

- **Source Code Structure**: All core modules organized under `src/` with logical separation
- **Utility Modules**: Common functionality grouped in `src/utils/` for reusability
- **Model Architecture**: ML models isolated in `src/models/` with clear interfaces
- **Script Organization**: Setup and utility scripts separated in `scripts/` directory
- **Clean Dependencies**: Modern uv-based dependency management with optional extras

### Password Transformations

Crackernaut implements various transformation strategies:

- **Character Substitution:** Replace letters with similar symbols
- **Case Modification:** Alter capitalization patterns
- **Numeric Manipulation:** Change numerical parts intelligently
- **Symbol Addition:** Insert special characters strategically
- **Pattern Recognition:** Apply common password creation patterns

### Machine Learning Models

- **🎯 Transformer Model (Primary):**  
  State-of-the-art architecture providing superior pattern recognition and batch processing capabilities
- **🔧 Legacy Models:**  
  MLP, RNN, BiLSTM models retained for backward compatibility and research purposes

### List Preparation System

- **📊 Chunked Processing:** Efficiently handles massive password datasets without memory overflow
- **🧠 Transformer Embeddings:** Generates sophisticated low-dimensional password representations
- **🎯 Clustering:** Uses Mini‑Batch K‑Means to group similar passwords for optimal training sets
- **✨ Representative Selection:** Intelligently chooses diverse samples for comprehensive training

### Processing Pipeline

1. **Input Processing**: Base password analysis and preparation
2. **Variant Generation**: Rule-based transformation pool creation
3. **ML Scoring**: Transformer-based variant likelihood assessment
4. **Intelligent Filtering**: Smart ranking and selection algorithms
5. **Optimized Output**: Top-scored variants with confidence metrics

## Development Environment

### GitHub Copilot Configuration
This workspace includes optimized GitHub Copilot configuration for ML security research:

- **Repository Instructions**: `.github/copilot-instructions.md` provides project-specific context
- **Workspace Settings**: `.vscode/settings.json` contains Copilot agent configuration
- **Privacy Protection**: `.copilotignore` excludes sensitive training data from indexing

The configuration ensures Copilot understands:
- Password security research context and ethical guidelines
- PyTorch/CUDA development patterns and error handling
- Async I/O patterns for large dataset processing
- Type safety and proper documentation standards

## Work in Progress

- List preparation module for organizing password datasets
- Updated training pipeline for transformer models
- Improved test coverage

## Disclaimer

Crackernaut is intended for ethical use only. Misuse of this tool for unauthorized access or malicious purposes is strictly prohibited.