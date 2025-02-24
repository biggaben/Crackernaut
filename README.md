# Crackernaut

**Purpose:**
Crackernaut is a password guessing utility designed to generate human-like password variants from a given base password. It combines rule-based transformations with machine learning to produce plausible password guesses that reflect common patterns humans use when creating passwords. The tool is intended for security researchers, penetration testers, or anyone needing to test password strength by generating realistic variants for analysis or cracking attempts.

**How It Works:**
Crackernaut operates through several key components, each implemented in separate Python modules:

1. **Main Script (`crackernaut.py`):**
   - **Functionality**: This is the primary script for generating password variants. It accepts a base password (via command-line argument or user prompt), generates variants, scores them using a machine learning model, and outputs the top variants to the console or a file.
   - **Process**:
     - Loads configuration from a JSON file (via `config_utils.py`) or uses defaults.
     - Generates variants using `generate_variants` from `variant_utils.py`.
     - Scores variants with an MLP model from `cuda_ml.py`, weighting modifications based on configuration.
     - Outputs the top-scoring variants, limited by user-specified parameters (`-n` for number, `-l` for length).

2. **Variant Generation (`variant_utils.py`):**
   - **Functionality**: Implements transformation functions to create password variants iteratively up to a specified chain depth.
   - **Transformations**:
     - Numeric increments (e.g., "Summer2020" → "Summer2021").
     - Symbol additions (e.g., "password" → "password!").
     - Capitalization tweaks (e.g., "password" → "Password").
     - Leet speak substitutions (e.g., "password" → "p@ssw0rd").
     - Shifts (e.g., "abc123" → "123abc").
     - Repetitions (e.g., "pass" → "pass!!").
     - Middle insertions (e.g., "password" → "pass@word").
   - **Mechanism**: Uses a queue-based approach to apply transformations up to `chain_depth`, ensuring variants stay within `max_length`.

3. **Machine Learning (`cuda_ml.py`):**
   - **Functionality**: Provides a CUDA-accelerated multi-layer perceptron (MLP) model to score variants and predict configuration adjustments.
   - **Model Details**:
     - Input: 8D feature vector extracted from base-variant pairs (e.g., numeric suffix length, symbol count, capitalization differences).
     - Output: 6D vector corresponding to modification types (Numeric, Symbol, Capitalization, Leet, Shift, Repetition).
     - Architecture: Four-layer MLP with configurable hidden dimensions and dropout.
   - **Training**: Supports training with `train_model`, using AdamW optimizer and Smooth L1 Loss on labeled data.
   - **Feature Extraction**: Computes features like leet character presence, symbol positions, and length differences.

4. **Training Script (`crackernaut_train.py`):**
   - **Functionality**: Trains the ML model and refines configuration weights in two modes:
     - **Bulk Training**: Processes a wordlist multiple times, generating variants and updating weights based on model predictions.
     - **Interactive Training**: Allows users to accept/reject variants, updating weights based on feedback.
   - **Hyperparameter Optimization**: Uses the Ax library to tune MLP parameters (hidden dimension, dropout) via Bayesian optimization.

5. **Configuration**:
   - Stored in `config.json`, with defaults including modification weights, chain depth, threshold, max length, and current base password.
   - Weights influence variant scoring and can be adjusted during training.

6. **Testing (`test_variants.py`):**
   - Contains basic tests to verify transformation functions and variant generation.

**Key Features:**
- GPU acceleration with CUDA for faster computation.
- Configurable transformation weights and generation parameters.
- Interactive and bulk training for model customization.
- Hyperparameter optimization for improved model performance.

**Use Case:**
Crackernaut is ideal for generating realistic password guesses for security testing, such as assessing the strength of password policies or simulating human behavior in password creation.

---

### README.md for GitHub

```markdown
# Crackernaut

Crackernaut is a password guessing utility that generates human-like password variants from a given base password. By combining rule-based transformations with a machine learning model, it produces plausible password guesses that reflect common human patterns. This tool is designed for security researchers, penetration testers, and anyone interested in testing password strength or simulating realistic password creation behaviors.

## Features

- **Human-Like Variants**: Generates passwords using transformations like numeric increments, symbol additions, capitalization changes, leet speak, shifts, repetitions, and middle insertions.
- **Machine Learning Scoring**: Uses a PyTorch-based MLP model to score variants based on their likelihood of human use.
- **GPU Acceleration**: Supports CUDA for faster computation on compatible hardware.
- **Configurable**: Adjust transformation weights, chain depth, and maximum length via a JSON configuration file.
- **Training Modes**:
  - Bulk training from a wordlist.
  - Interactive training with user feedback.
- **Hyperparameter Tuning**: Optimizes the ML model using Bayesian optimization (Ax library).
- **Flexible Output**: Outputs variants to the console or a file, with options to limit quantity and length.

## Requirements

### Software
- Python 3.8 or higher
- PyTorch 1.9 or higher (with CUDA support for GPU acceleration)
- Dependencies listed in `requirements.txt`:
  - numpy>=2.2.3
  - tqdm>=4.67.1
  - pycuda>=2025.1
  - ax-platform>=0.4.1

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

Run `crackernaut.py` to generate variants from a base password:

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

Use `crackernaut_train.py` to train the ML model and refine configuration:

#### Bulk Training
Train on a wordlist (one password per line):

```bash
python crackernaut_train.py --wordlist <wordlist_file> [-t <iterations>]
```

- `--wordlist`: Path to the wordlist file.
- `-t, --times`: Number of training iterations (default: 1).

**Example**:
```bash
python crackernaut_train.py --wordlist common_1k.txt -t 3
```

#### Interactive Training
Fine-tune the model with user feedback:

```bash
python crackernaut_train.py --interactive [-b <base_password>] [-a <alternatives>]
```

- `--interactive`: Enables interactive mode.
- `-b, --base`: Optional base password (defaults to config's `current_base`).
- `-a, --alternatives`: Number of variants to show per round (default: 5).

**Example**:
```bash
python crackernaut_train.py --interactive -b "Summer2023"
```

In interactive mode:
- View a sample of variants.
- Accept (e.g., "1,3" for indices 1 and 3), reject (`r`), or adjust settings (`k` for base, `c` for config).
- Save and exit with `save`, or exit without saving with `exit`.

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
    "current_base": "Password123!"
}
```

- **`modification_weights`**: Influence scoring for each transformation type.
- **`chain_depth`**: Maximum number of transformations applied iteratively.
- **`max_length`**: Default maximum variant length.
- **`current_base`**: Base password for interactive training.

Edit this file to customize behavior, or reset to defaults during interactive training.

## How It Works

1. **Variant Generation**:
   - Applies transformations (e.g., "password" → "P@ssw0rd!") up to `chain_depth`.
   - Ensures variants stay within `max_length`.

2. **Scoring**:
   - Extracts 8D features (e.g., symbol count, leet presence) from each variant.
   - Uses the MLP model to predict a 6D modification vector.
   - Scores variants by weighting predictions with config weights.

3. **Training**:
   - **Bulk**: Generates variants from a wordlist, predicts adjustments, and updates weights.
   - **Interactive**: Updates weights based on user-accepted variants.
   - **Optimization**: Tunes MLP hidden dimension and dropout using Ax.

4. **Output**:
   - Returns top-scoring variants, mimicking human password patterns.

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

This project is licensed under the MIT License. See the `LICENSE` file for details.

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

Crackernaut is offered "as is," with no guarantees or promises about its performance or suitability for any particular task. If it doesn’t work as you expect, or if it leads to unintended results due to your use, that’s on you—not me.

**A Word on Ethics**

I strongly urge you to use Crackernaut ethically. Respect the privacy, security, and rights of others. Only use it where you’ve been given the green light to test or experiment. Ethical use isn’t just a suggestion—it’s the only way this tool aligns with its intended purpose.

**Final Note**

By using Crackernaut, you acknowledge that you’ve read this statement and understand your responsibilities. I’ve done my part by creating a tool for learning and legitimate testing. What you do with it is up to you—and I’m not accountable for it.
