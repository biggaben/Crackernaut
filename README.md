# Crackernaut

A tool for password variant generation and scoring using machine learning.

## Recent Updates

- **Transformer Model for Variant Scoring:**  
  Added a lightweight transformer-based model to generate richer embeddings and score password variants more accurately.
- **Enhanced Pipeline Integration:**  
  Updated the main pipeline to support transformer-based scoring, ensuring efficient GPU utilization and faster processing.
- **Smart List Preparation Module:**  
  Implemented a new list-preparer module that processes massive password datasets asynchronously, clusters similar passwords using Mini‑Batch K‑Means, and selects representative samples for optimized training.
- **Improved Batch Processing:**  
  Enhanced batch processing of password variants for better performance on large datasets.

## Purpose

Crackernaut is a sophisticated password guessing utility designed to generate human-like password variants from a given base password. It combines rule-based transformations with machine learning to produce plausible password guesses that reflect common patterns humans use when creating passwords. This tool is intended for security researchers, penetration testers, and anyone needing to test password strength by generating realistic variants for analysis or cracking attempts.

## Use Case

Crackernaut is ideal for generating realistic password guesses for security testing, such as assessing the strength of password policies or simulating human behavior in password creation.

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

- Python 3.6+
- PyTorch
- CUDA (optional, for GPU acceleration)

Dependencies are listed in [`requirements.txt`](requirements.txt):
- numpy>=2.2.3
- python-Levenshtein>=0.26.1
- scikit-learn>=1.5.0 (for clustering in list preparation)

### Hardware

- **Recommended:** CUDA-capable GPU (e.g., Nvidia RTX 3090) for optimal performance.
- **Minimum:** Any CPU (runs without GPU support, though slower).

## Installation

1. Clone this repository.
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
   Ensure you have PyTorch installed with CUDA support (if using GPU).

## Usage

### Basic Usage
Run Crackernaut to generate variants from a base password:
```bash
python crackernaut.py --password "mypassword" --model transformer
```

### Command Line Options
- `--password, -p`: Base password to analyze.
- `--config, -c`: Path to configuration file (default: config.json).
- `--depth, -d`: Chain depth for variant generation.
- `--model, -m`: Model type: transformer, rnn, bilstm, mlp (default: transformer).

### Generating Password Variants
```bash
python crackernaut.py -p <base_password> [-l <max_length>] [-n <output_count>] [-o <output_file>]
```
- `-p, --password`: Base password (prompts if not provided).
- `-l, --length`: Maximum length of variants.
- `-n, --number`: Number of top variants to output.
- `-o, --output`: File to save variants.

Example:
```bash
python crackernaut.py -p "Summer2023" -l 15 -n 10 -o variants.txt
```

### Training the Model
Use `crackernaut_train.py` to train the ML model and refine configuration.

#### Bulk Training
Train on a wordlist (one password per line):
```bash
python crackernaut_train.py --wordlist <wordlist_file> [-t <iterations>] [--model <model_type>]
```
Example:
```bash
python crackernaut_train.py --wordlist rockyou.txt --times 5 --model bilstm
```

#### Intelligent Dataset Preparation
Process large wordlists into optimized training sets:
```bash
python crackernaut_train.py --prepare --lp-dataset <path_to_dataset> [--clusters <num_clusters>] [--lp-chunk-size <size>] [--lp-output <output_dir>]
```
Example:
```bash
python crackernaut_train.py --prepare --lp-dataset breach_compilation.txt --clusters 20000 --lp-chunk-size 2000000
```

#### Interactive Training
Fine-tune the model with interactive feedback:
```bash
python crackernaut_train.py --interactive
```

### Configuration
Customize Crackernaut via the `config.json` file. Key options include:
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

### Usage
1. Preprocess passwords: `python list_preparer.py --input passwords.txt`
2. Train the model: `python crackernaut_train.py --config config.json`
3. Run interactively: `python crackernaut_train.py --interactive`

## Work in Progress

- List preparation module for organizing password datasets
- Updated training pipeline for transformer models
- Improved test coverage

## Technical Architecture

### Password Transformations
Crackernaut implements various transformation strategies:
- **Character Substitution:** Replace letters with similar symbols.
- **Case Modification:** Alter capitalization.
- **Numeric Manipulation:** Change numerical parts.
- **Symbol Addition:** Insert special characters.
- **Pattern Recognition:** Apply common creation patterns.

### Machine Learning Models
- **Transformer Model (NEW):**  
  Provides superior pattern recognition and batch processing.
- **Legacy Models:**  
  MLP, RNN, BiLSTM are retained for backward compatibility.

### List Preparation System
- **Chunked Processing:** Efficiently handles massive datasets.
- **Transformer Embeddings:** Generates low-dimensional embeddings.
- **Clustering:** Uses Mini‑Batch K‑Means to group similar passwords.
- **Representative Selection:** Chooses diverse samples for training.

### Processing Pipeline
- Base password input
- Generation of initial variant pool
- ML-based scoring of variants
- Filtering and ranking
- Output of top variants

### Training Methods
- Supervised Learning from known password pairs.
- Self-Supervised Learning from pattern mining.
- Interactive Learning via user feedback.
- Intelligent Dataset Reduction via clustering.

## Disclaimer
Crackernaut is provided for educational and authorized security testing purposes only. Use responsibly and legally.
