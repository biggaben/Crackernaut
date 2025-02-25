# Implementation Plan for Crackernaut

This document outlines the updated steps for integrating the new transformer‐based variant scoring model and list preparation module into Crackernaut. Each file is listed with its current status and the actions required to bring the project to completion.

---

## Overall Objectives

- **Transformer-Based Variant Scoring (DONE):**  
  Replace legacy MLP/RNN/BiLSTM models with a lightweight transformer model that processes raw character sequences to produce richer embeddings and scores for password variants.  
  - *Status:* Transformer model is implemented in `transformer_model.py` and integrated into `crackernaut.py`.

- **List Preparation Module (PENDING):**  
  Fully implement `list_preparer.py` to:
  - Asynchronously load huge password datasets (with Unicode and variable-length support).
  - Generate embeddings using a lightweight transformer encoder.
  - Cluster passwords using Mini‑Batch K‑Means.
  - Select representative samples from each cluster for building high-quality training sets.
  - *Status:* Most functions are present; final testing, error handling, and representative selection need to be verified and finalized.

- **Configuration, Training, and Integration (PARTIALLY COMPLETE):**  
  Update configuration files and training scripts so that the new transformer and list-preparation functionalities integrate seamlessly with legacy workflows. Legacy functions are retained (and clearly marked as such) to ensure backward compatibility.  
  - *Status:*  
    - `config_utils.py` has been updated with the new transformer and list preparation keys (**DONE**).  
    - `crackernaut_train.py` has been fully updated to support transformer training and list preparation mode (**DONE**).

- **Testing & Documentation (DONE):**  
  Unit tests in `test_variants.py` have been updated to cover new functionality (Unicode support, transformer output shape, gradient checkpointing). Documentation in README.md and Instructions.txt has been updated with detailed usage, ethical guidelines, and technical architecture details.

---

## File-by-File Detailed Plan

### 1. async_utils.py (DONE)
**Purpose:**  
Provides asynchronous file I/O for loading and saving large datasets.  
**Status:**  
- Functions such as `async_load_passwords` and `async_save_results` are complete.  
**Action:**  
- Use these functions in `list_preparer.py`.

---

### 2. config_utils.py (DONE)
**Purpose:**  
Manages configuration settings for both legacy and new transformer/list preparation workflows.  
**Modifications Completed:**  
- Added new keys for transformer parameters:
  - `"model_embed_dim"`, `"model_num_heads"`, `"model_num_layers"`, `"model_hidden_dim"`, `"model_dropout"`.
- Added keys for list preparer options:
  - `"lp_chunk_size"`, `"lp_output_dir"`.
  
**Example Updated DEFAULT_CONFIG:**
```python
DEFAULT_CONFIG = {
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
    "model_type": "transformer",
    "model_embed_dim": 64,
    "model_num_heads": 4,
    "model_num_layers": 3,
    "model_hidden_dim": 128,
    "model_dropout": 0.2,
    "lp_chunk_size": 1000000,
    "lp_output_dir": "clusters"
}
```
- Load/save functions remain unchanged.

---

### 3. crackernaut.py (DONE)
**Purpose:**  
Main production script for generating and scoring password variants.  
**Modifications Completed:**  
- Imported the new transformer model from `transformer_model.py`.
- Updated the model-loading logic to instantiate the transformer model when `config["model_type"] == "transformer"`.
- Legacy variant generation via `variant_utils.py` is retained.
- Scoring calls now use the transformer model’s forward() method.

---

### 4. crackernaut_train.py (DONE)
**Purpose:**  
Training script for the ML model; supports bulk, interactive, and list preparation training modes.  
**Modifications Completed:**  
- Added command-line options: `--prepare`, `--lp-dataset`, `--lp-output`, `--lp-chunk-size`.
- Integrated list preparation mode by calling `run_preparation` from `list_preparer.py` when `--prepare` is specified.
- Updated model selection to use the transformer model if specified in the config.
- Retained legacy training modes.

---

### 5. cuda_ml.py (DONE as Legacy)
**Purpose:**  
Implements legacy models (MLP, RNN, BiLSTM) for password scoring.  
**Modifications:**  
- A warning comment has been added indicating that this module is legacy and that transformer-based scoring is now in `transformer_model.py`.

---

### 6. distributed_training.py (DONE)
**Purpose:**  
Handles distributed training using PyTorch DDP.  
**Modifications:**  
- Reviewed and confirmed compatibility with transformer models.
- No major code changes required.

---

### 7. performance_utils.py (DONE)
**Purpose:**  
Provides helper functions for resource management and batch size estimation.  
**Modifications:**  
- Updated comments to note that for transformer models, the `feature_vector_size` should be based on the embedding dimension.

---

### 8. test_variants.py (DONE)
**Purpose:**  
Unit tests for variant generation and scoring functionality.  
**Modifications Completed:**  
- Added tests for Unicode support and padding in `text_to_tensor`.
- Tested transformer model output shape and functionality.
- Tested gradient checkpointing and other new features.
- Retained legacy tests for backward compatibility.

---

### 9. variant_utils.py (DONE)
**Purpose:**  
Generates password variants using transformation rules.  
**Modifications:**  
- No changes required; a comment was added to indicate that scoring is now handled by the transformer model in `transformer_model.py`.

---

### 10. list_preparer.py (PENDING)
**Purpose:**  
Prepares and structures massive password datasets by generating embeddings, clustering similar passwords, and selecting representative samples.  
**Modifications Required:**  
- Finalize the asynchronous loading, embedding extraction, clustering, and representative selection functions.
- Remove outdated placeholders, add robust error handling, and ensure comprehensive testing for Unicode and variable-length passwords.
  
**Example Updated Pseudocode:**
```python
async def load_password_chunks(file_path: str, chunk_size: int = 1000000):
    seen = set()
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        async for line in f:
            pwd = line.strip()
            if len(pwd) >= 4 and pwd not in seen:
                chunk.append(pwd)
                seen.add(pwd)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

class PasswordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, mask=None):
        emb = self.embedding(x).transpose(0, 1)
        out = self.encoder(emb, src_key_padding_mask=mask)
        return out.mean(dim=0)

def extract_embeddings(model, password_list, batch_size=1024, device="cuda", max_length=20):
    model.eval()
    embeddings = []
    for i in range(0, len(password_list), batch_size):
        batch = password_list[i:i+batch_size]
        tensors = text_to_tensor(batch, max_length=max_length, device=device)
        masks = (tensors != 0).to(device)
        with torch.no_grad(), autocast():
            emb = model(tensors, mask=~masks)
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)
```
*Note: Final testing and error handling improvements are still required.*

---

### 11. transformer_model.py (DONE)
**Purpose:**  
Implements the new lightweight transformer model for password scoring.  
**Status:**  
- Fully implemented with positional encoding, forward pass, batch scoring, saving/loading functionality, and helper methods.

---

## Final Status Summary

- **Transformer-Based Variant Scoring:**  
  - Transformer model (`transformer_model.py`): **DONE**  
  - Integration in `crackernaut.py`: **DONE**

- **List Preparation Module:**  
  - Asynchronous loading and filtering: **DONE** (via async_utils.py and implemented functions in list_preparer.py)  
  - Transformer-based embedding extraction, clustering, and representative selection: **PENDING** (final testing and error handling required)

- **Configuration Updates:**  
  - New transformer and list preparation keys in `config_utils.py`: **DONE**

- **Training and Integration:**  
  - Updates in `crackernaut_train.py` for transformer training and list preparation mode: **DONE**

- **Legacy Code (cuda_ml.py):**  
  - Remains available with warning comments: **DONE**

- **Distributed Training & Performance Utilities:**  
  - `distributed_training.py` and `performance_utils.py`: **DONE**

- **Testing:**  
  - Unit tests in `test_variants.py` updated to cover new functionality: **DONE**

- **Documentation:**  
  - README.md and Instructions.txt updated with detailed sections on recent updates, ethical guidelines, usage instructions, and technical architecture: **DONE**

---

## Next Steps

1. **Finalize `list_preparer.py`:**  
   - Complete and thoroughly test embedding extraction, clustering, and representative selection.  
2. **Expand Integration Testing:**  
   - Conduct further integration tests, especially for list preparation and downstream usage in training.  
3. **Review and Finalize Documentation:**  
   - Confirm that all usage instructions and ethical guidelines are fully up to date.

By following this updated plan, we can ensure that all components of Crackernaut are modernized, outdated code and placeholders are removed, and the system is robustly integrated with the new transformer-based functionality.