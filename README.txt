# Crackernaut

**Purpose:**
Crackernaut is a sophisticated password guessing utility designed to generate human-like password variants from a given base password. It combines rule-based transformations with machine learning to produce plausible password guesses that reflect common patterns humans use when creating passwords. The tool is intended for security researchers, penetration testers, or anyone needing to test password strength by generating realistic variants for analysis or cracking attempts.

**Use Case:**
Crackernaut is ideal for generating realistic password guesses for security testing, such as assessing the strength of password policies or simulating human behavior in password creation.

## Features

- **Human-Like Variants**: Generates passwords using transformations like numeric increments, symbol additions, capitalization changes, leet speak, shifts, repetitions, and middle insertions.
- **Machine Learning Scoring**: Uses PyTorch-based models (MLP, RNN, BiLSTM) to score variants based on their likelihood of human use.
- **GPU Acceleration**: Leverages CUDA for faster computation on compatible hardware with automatic device detection.
- **Configurable**: Adjust transformation weights, chain depth, and maximum length via a JSON configuration file.
- **Multi-Processing**: Employs parallel processing for variant generation and a producer-consumer pattern for efficient processing.
- **Training Modes**:
  - Bulk training from wordlists with automated learning
  - Interactive training with user feedback
  - Self-supervised learning from password pattern mining
- **Hyperparameter Tuning**: Optimizes ML models using Bayesian optimization with the Ax library.
- **Distributed Training**: Supports distributed data parallelism across multiple GPUs for faster training.
- **Asynchronous I/O**: Uses asynchronous file operations for efficient data handling.
- **Flexible Output**: Outputs variants to the console or a file, with options to limit quantity and length.

## Requirements

### Software
- Python 3.8 or higher
- PyTorch 1.9 or higher (with CUDA support for GPU acceleration)
- Dependencies listed in [`requirements.txt`](requirements.txt):
  - numpy>=2.2.3
  - tqdm>=4.67.1
  - pycuda>=2025.1
  - ax-platform>=0.4.1
  - python-Levenshtein>=0.26.1

### Hardware
- **Recommended**: CUDA-capable GPU for optimal performance.
- **Minimum**: Any CPU (runs without GPU support, though slower).

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/crackernaut.git
   cd crackernaut
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify PyTorch with CUDA** (if using GPU):
   ```python
   import torch
   print(torch.cuda.is_available())  # Should return True if CUDA is detected
   ```

## Usage

### Generating Password Variants

Run [`crackernaut.py`](crackernaut.py) to generate variants from a base password:

```bash
python crackernaut.py -p <base_password> [-l <max_length>] [-n <output_count>] [-o <output_file>]
```

- `-p, --password`: Base password (prompts if not provided).
- `-l, --length`: Maximum length of variants (overrides config default).
- `-n, --number`: Number of top variants to output.
- `-o, --output`: File to save variants (overwrites or appends based on user choice).

**Example**:
```bash
python crackernaut.py -p "Summer2023" -l 15 -n 10 -o variants.txt
```

This generates up to 10 variants from "Summer2023", each up to 15 characters, and saves them to `variants.txt`.

### Training the Model

Use [`crackernaut_train.py`](crackernaut_train.py) to train the ML model and refine configuration:

#### Bulk Training
Train on a wordlist (one password per line):

```bash
python crackernaut_train.py --wordlist <wordlist_file> [-t <iterations>] [--model <model_type>]
```

- `--wordlist`: Path to the wordlist file.
- `-t, --times`: Number of training iterations (default: 1).
- `--model`: Model architecture to use (mlp, rnn, or bilstm, default: rnn).
- `--supervised`: Use self-supervised learning to extract patterns from the wordlist.

**Example**:
```bash
python crackernaut_train.py --wordlist common_1k.txt -t 3 --model bilstm
```

#### Interactive Training
Fine-tune the model with user feedback:

```bash
python crackernaut_train.py --interactive [-b <base_password>] [-a <alternatives>]
```

- `--interactive`: Enables interactive mode.
- `-b, --base`: Optional base password (defaults to config's `current_base`).
- `-a, --alternatives`: Number of variants to show per round (default: 5).
- `--learning-rate`: Override the default learning rate.

**Example**:
```bash
python crackernaut_train.py --interactive -b "Summer2023" -a 8
```

In interactive mode:
- View a sample of variants.
- Accept (e.g., "1,3" for indices 1 and 3), reject (`r`), or adjust settings (`k` for base, `c` for config).
- Save and exit with `save`, or exit without saving with `exit`.

### Advanced Usage

#### Distributed Training
For large datasets on multi-GPU systems:

```bash
python crackernaut_train.py --wordlist <large_wordlist> --distributed --gpus <num_gpus>
```

#### Performance Optimization
Use the [`performance_utils.py`](performance_utils.py) module to optimize batch sizes and parallelism strategies:

```python
from performance_utils import optimize_batch_size, choose_parallelism_strategy, get_device_info

device_info = get_device_info()
batch_size = optimize_batch_size(model_size_mb=150, feature_vector_size=8)
strategy = choose_parallelism_strategy("inference", device_info)
```

## Configuration

The `config.json` file stores settings (defaults if not present):

```json
{
    "modification_weights": {
        "Numeric": 1.0,
        "Symbol": 1.0,
        "Capitalization": 1.0,
        "Leet": 1.0,
        "Shift": 1.0,
        "Repetition": 1.0
    },
    "chain_depth": 2,
    "threshold": 0.5,
    "max_length": 20,
    "current_base": "Password123!",
    "learning_rate": 0.01,
    "model_type": "rnn",
    "model_embed_dim": 32,
    "model_hidden_dim": 64,
    "model_num_layers": 2,
    "model_dropout": 0.2
}
```

- **`modification_weights`**: Influence scoring for each transformation type.
- **`chain_depth`**: Maximum number of transformations applied iteratively.
- **`max_length`**: Default maximum variant length.
- **`current_base`**: Base password for interactive training.
- **`learning_rate`**: Learning rate for model training.
- **`model_type`**: Type of neural network model to use.
- **Model architecture parameters**: Configure the neural network architecture.

Edit this file to customize behavior, or reset to defaults during interactive training.

## Technical Architecture

### Password Transformations

Crackernaut uses the following transformation techniques defined in [`variant_utils.py`](variant_utils.py):

1. **Numeric Increment**: Adds or increments numbers (e.g., "password123" → "password124")
2. **Symbol Addition**: Adds special characters (e.g., "password" → "password!")
3. **Capitalization**: Changes letter case (e.g., "password" → "Password")
4. **Leet Substitution**: Replaces letters with similar-looking symbols (e.g., "password" → "p@ssw0rd")
5. **Shift Variants**: Moves numeric portions (e.g., "abc123" → "123abc")
6. **Middle Insertion**: Inserts characters in the middle (e.g., "password" → "pass$word")
7. **Repetition**: Repeats characters or adds duplicates (e.g., "password" → "passwordd")

Transformations can be applied in chains (combinations) up to the configured depth.

### Machine Learning Models

Crackernaut supports three ML model architectures in [`cuda_ml.py`](cuda_ml.py):

1. **MLP (Multi-Layer Perceptron)**:
   - Feedforward neural network for feature-based prediction
   - Input: 8D feature vector extracted from password pairs
   - Output: 6D vector of modification preference scores

2. **PasswordRNN**:
   - Character-level RNN with LSTM layers
   - Processes passwords as sequences of character indices
   - Better captures sequential patterns in passwords

3. **PasswordBiLSTM**:
   - Bidirectional LSTM with attention mechanism
   - Improved understanding of character relationships
   - Better for complex password patterns

### Processing Pipeline

The [`crackernaut.py`](crackernaut.py) script implements a multi-threaded pipeline:

1. **Variant Generation**: Producer threads generate password variants.
2. **Feature Extraction**: Extract features from base-variant pairs.
3. **Scoring**: Consumer threads score variants using the ML model.
4. **Ranking**: Variants are sorted by score for output.

This producer-consumer architecture allows efficient processing of large numbers of variants.

### Training Methods

1. **Bulk Training** ([`crackernaut_train.py`](crackernaut_train.py)):
   - Processes wordlists to generate variant training pairs
   - Updates model weights based on batch processing
   - Supports hyperparameter optimization

2. **Interactive Training**:
   - Shows sample variants to the user
   - Updates weights based on user selections
   - Allows fine-tuning of model behavior

3. **Self-Supervised Learning**:
   - Automatically extracts likely password variants from wordlists
   - Uses password similarity and pattern mining
   - Identifies year patterns, incremental patterns, and clustering

4. **Distributed Training** ([`distributed_training.py`](distributed_training.py)):
   - Supports training across multiple GPUs
   - Uses PyTorch's DistributedDataParallel
   - Scales learning rates and batch sizes automatically

## Performance Considerations

1. **Memory Optimization**:
   - Batch processing for large wordlists
   - Optimal batch size calculation based on available GPU memory

2. **Parallelism**:
   - Multi-processing for variant generation
   - Producer-consumer pattern for pipeline efficiency
   - Support for multi-GPU training

3. **I/O Efficiency**:
   - Asynchronous file operations ([`async_utils.py`](async_utils.py))
   - Streaming processing for large files

4. **GPU Acceleration**:
   - CUDA support for tensor operations
   - Automatic mixed precision training
   - Device-aware tensor allocation

## Output

Variants are ranked by ML scores, reflecting their plausibility as human-created passwords. Example output for "Summer2023":
```
Summer2024!
SUMMER2023
Summer2023@
Summ3r2023
2023Summer
```

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-name`).
5. Open a pull request.

Please open an issue first for significant changes or feature requests.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Acknowledgments

- Built with PyTorch for ML and CUDA support.
- Uses Ax for hyperparameter optimization.
- Inspired by real-world password generation patterns.

---
Happy password guessing! For questions, open an issue on GitHub.
---

## Non-Disclaimer Statement for Crackernaut

**Purpose of Crackernaut**

Crackernaut is a tool created **exclusively for educational purposes and legitimate, authorized penetration testing**. Its sole intent is to assist security researchers, ethical hackers, and individuals with explicit permission to test password strength and security practices in controlled, legal environments. This tool is designed to promote learning about password security and to support authorized security assessments.

**Educational and Legitimate Use Only**

This tool is provided with the clear expectation that it will be used responsibly and legally. It is meant to help users understand password creation patterns and evaluate password policies in scenarios where they have been granted permission to do so. Any use beyond these purposes—especially for malicious or unauthorized activities—is explicitly outside the scope of its intended design.

**User Responsibility**

As the user of Crackernaut, **you are entirely responsible for how you choose to use this tool**. Your actions, decisions, and their consequences rest solely with you. You must ensure that your use of Crackernaut complies with all applicable laws, regulations, and ethical standards in your jurisdiction. This includes obtaining explicit authorization before conducting any penetration testing or security assessments.

**No Developer Responsibility**

I, as the developer of Crackernaut, **am not responsible under any circumstances for the actions of users**. Whether you use this tool as intended or misuse it for malicious, illegal, or unauthorized purposes, I bear no liability for your conduct or its outcomes. Any consequences—legal, financial, or otherwise—arising from your use of Crackernaut are yours alone to bear.

**Prohibited Malicious Use**

Let me be clear: Crackernaut is **not intended for malicious purposes**. Using it to harm others, gain unauthorized access to systems or networks, or engage in any illegal activity is strictly against its purpose. I do not support, endorse, or condone such actions in any way.

**Provided "As Is"**

Crackernaut is offered "as is," with no guarantees or promises about its performance or suitability for any particular task. If it doesn't work as you expect, or if it leads to unintended results due to your use, that's on you—not me.

**A Word on Ethics**

I strongly urge you to use Crackernaut ethically. Respect the privacy, security, and rights of others. Only use it where you've been given the green light to test or experiment. Ethical use isn't just a suggestion—it's the only way this tool aligns with its intended purpose.

**Final Note**

By using Crackernaut, you acknowledge that you've read this statement and understand your responsibilities. I've done my part by creating a tool for learning and legitimate testing. What you do with it is up to you—and I'm not accountable for it.
