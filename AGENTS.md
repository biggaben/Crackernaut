# Crackernaut - AI Agents Configuration

## Project Overview

Crackernaut is a sophisticated password security research tool that generates human-like password variants using machine learning. This project combines rule-based transformations with ML models including transformers, RNNs, and MLPs for password variant scoring and generation.

## Available AI Agents

### 1. Password Variant Generator Agent

**Role**: Generate realistic password variants using ML models
**Capabilities**:

- Transform base passwords into human-like variants
- Apply rule-based and ML-driven transformations
- Score variant likelihood using transformer models
- Support batch processing for large datasets

**Usage Context**:

```python
# Generate variants for penetration testing
variants = generator.generate_variants("password123", num_variants=100)
```

### 2. ML Model Training Agent

**Role**: Train and optimize password prediction models
**Capabilities**:

- Train transformer, RNN, and MLP models on password datasets
- Perform hyperparameter optimization using Bayesian methods
- Handle distributed multi-GPU training
- Implement proper train/validation/test splits

**Usage Context**:

```python
# Train new transformer model on dataset
trainer.train_model(config_path="config.json", dataset_path="trainingdata/")
```

### 3. Dataset Processing Agent

**Role**: Prepare and optimize password datasets for training
**Capabilities**:

- Clean and deduplicate password datasets
- Perform Mini-Batch K-Means clustering for optimization
- Handle asynchronous file I/O for large datasets
- Generate training/validation splits

**Usage Context**:

```python
# Process raw password list for training
processor.prepare_dataset("raw_passwords.txt", output_dir="trainingdata/")
```

### 4. Security Analysis Agent
    
**Role**: Analyze password strength and vulnerability patterns
**Capabilities**:

- Evaluate password complexity and predictability
- Identify common transformation patterns
- Generate security reports and recommendations
- Support authorized penetration testing workflows

**Usage Context**:

```python
# Analyze password strength metrics
analysis = analyzer.evaluate_passwords(password_list, include_variants=True)
```

## Technical Requirements

### Environment Setup

- **Python**: 3.8+ with type hints and async/await support
- **Package Manager**: uv for fast, reliable dependency management
- **ML Framework**: PyTorch with CUDA 12.1 GPU acceleration
- **Key Dependencies**: transformers, scikit-learn, aiofiles, NumPy

### Hardware Requirements

- **GPU**: CUDA 12.1+ compatible GPU (tested on RTX 3090)
- **CUDA**: CUDA 12.1 or later with compute capability 7.0+
- **Memory**: 16GB+ RAM for large dataset processing (24GB+ VRAM recommended)
- **Storage**: SSD recommended for fast I/O operations

### CUDA Setup

The project requires PyTorch with CUDA 12.1 support. Installation is handled automatically via uv:

```bash
# Install with CUDA support
uv sync --extra cuda

# Verify CUDA installation
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Security Considerations

- Never log actual passwords in plaintext
- Validate and sanitize all password inputs
- Use secure coding practices for sensitive research data
- Follow responsible disclosure principles
- Ensure all testing is authorized and ethical

## Configuration

### Model Configuration (config.json)

```json
{
  "model_type": "transformer",
  "batch_size": 32,
  "learning_rate": 0.001,
  "epochs": 100,
  "gpu_enabled": true,
  "distributed": false
}
```

### Agent Initialization

```python
# Initialize with configuration
config = load_config("config.json")
agent = PasswordVariantAgent(config)
```

## File Structure Context

```code
Crackernaut/
├── models/           # ML model implementations
├── trainingdata/     # Password datasets (gitignored)
├── clusters/         # Processed clustering data
├── crackernaut.py    # Main application entry point
├── crackernaut_train.py  # Training pipeline
├── list_preparer.py  # Dataset processing utilities
├── config.json       # Model and processing configuration
├── pyproject.toml    # Project dependencies and metadata
└── AGENTS.md         # This file
```

## Usage Patterns

### Batch Processing

```python
# Process large datasets efficiently
async with BatchProcessor(config) as processor:
    results = await processor.process_file("large_dataset.txt")
```

### GPU Memory Management

```python
# Proper CUDA memory handling with device detection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Clear cache when needed (important for large models)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    
# Monitor GPU memory usage
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"Memory Allocated: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
```

### Asynchronous I/O

```python
# Handle large files asynchronously
async with aiofiles.open("passwords.txt", "r") as f:
    async for line in f:
        yield process_password(line.strip())
```

## Performance Optimization

- Use batch processing for large datasets to avoid memory issues
- Implement producer-consumer patterns for efficient data flow
- Clear GPU cache appropriately during training loops
- Use generators for memory-efficient dataset iteration
- Monitor memory usage during ML operations

## Error Handling

```python
try:
    variants = agent.generate_variants(password)
except PasswordValidationError as e:
    logger.error(f"Invalid password input: {e}")
except RuntimeError as e:
    if "CUDA" in str(e):
        logger.error(f"CUDA error: {e}")
        # Fallback to CPU processing
        device = torch.device("cpu")
        model = model.to(device)
    else:
        logger.error(f"Runtime error: {e}")
except torch.cuda.OutOfMemoryError as e:
    logger.error(f"GPU memory insufficient: {e}")
    # Clear cache and retry with smaller batch
    torch.cuda.empty_cache()
```

## Logging and Monitoring

- Implement comprehensive logging without exposing sensitive data
- Use tqdm for progress bars on long-running operations
- Monitor GPU utilization and memory usage
- Track model performance metrics during training

## Research Ethics

This tool is designed for legitimate security research and authorized penetration testing. All users must:

- Obtain proper authorization before testing
- Follow responsible disclosure principles
- Use generated variants only for legitimate security purposes
- Respect privacy and legal boundaries

## Getting Started

1. **Setup Environment**:

```bash
   # Install base dependencies
   uv sync
   
   # Install with CUDA support (recommended)
   uv sync --extra cuda
   
   # Verify CUDA setup
   uv run python scripts/check_gpu.py
```

2. **Configure Models**:

```bash
   # Edit config.json for your requirements
   # Ensure gpu_enabled: true for CUDA acceleration
```

3. **Prepare Dataset**:

```bash
   uv run python src/list_preparer.py --input-dir trainingdata --output-dir clusters
```

4. **Train Model**:

```bash
    uv run python crackernaut_train.py --config config.json --epochs 100
```

5. **Generate Variants**:

```bash
    uv run python crackernaut.py --password "example123" --config config.json
```

## Development Workflow

### Available VS Code Tasks

The project includes pre-configured VS Code tasks for common operations:

- **Install Dependencies**: `uv sync` and `uv sync --extra cuda`
- **Run Crackernaut**: Execute main application with input prompts
- **Train Model**: Run training pipeline with configurable epochs
- **Prepare Training Data**: Process raw password datasets
- **Run Tests**: Execute pytest test suite
- **Check GPU Status**: Verify CUDA and PyTorch installation
- **Format Code**: Apply Black formatting with 88-character limit
- **Lint Code**: Run flake8 with project-specific settings

### CUDA Verification Commands

```bash
# Check CUDA availability
uv run python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"

# Detailed GPU information
uv run python scripts/check_gpu.py

# Run CUDA test
uv run python simple_cuda_test.py
```

## Support and Documentation

For detailed documentation on specific agents and their capabilities, refer to the individual module docstrings and the project's main documentation.
