#!/usr/bin/env python3
"""
list_preparer.py

This module implements the list‐preparing model for Crackernaut.
It processes a massive password dataset in memory‐efficient chunks,
generates low‑dimensional embeddings using a lightweight transformer encoder,
clusters the embeddings using Mini‑Batch K‑Means, and selects representative
passwords from each cluster to create a structured, diverse training set.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
import asyncio
import aiofiles
import time
from tqdm import tqdm
from sklearn.cluster import MiniBatchKMeans
from transformer_model import PasswordTransformer
from async_utils import async_save_passwords, async_load_passwords, async_save_results
from torch.cuda.amp import autocast

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('list_preparer')

# Convert passwords to tensors with padding and Unicode support
def text_to_tensor(passwords, max_length=20, device="cuda", vocab_size=256):
    batch = []
    for pwd in passwords:
        indices = [ord(c) % vocab_size for c in pwd[:max_length]]  # Map characters to indices
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # Pad with zeros
        batch.append(indices)
    return torch.tensor(batch, dtype=torch.long, device=device)

# Transformer model for embedding generation
class PasswordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=32, num_heads=4, num_layers=2, hidden_dim=64, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, mask=None):
        emb = self.embedding(x).transpose(0, 1)  # Shape: (seq_len, batch, embed_dim)
        out = self.encoder(emb, src_key_padding_mask=mask)  # Shape: (seq_len, batch, embed_dim)
        return out.mean(dim=0)  # Mean pooling: (batch, embed_dim)

# Extract embeddings from password list
def extract_embeddings(model, password_list, batch_size=1024, device="cuda", max_length=20):
    model.eval()
    embeddings = []
    for i in range(0, len(password_list), batch_size):
        batch = password_list[i:i + batch_size]
        tensors = text_to_tensor(batch, max_length=max_length, device=device)
        masks = (tensors != 0).to(device)  # True for non-padding tokens
        with torch.no_grad(), autocast():  # Mixed precision for efficiency
            emb = model(tensors, mask=~masks)  # ~masks to ignore padding
        embeddings.append(emb.cpu().numpy())
    return np.vstack(embeddings)

# Cluster passwords using Mini-Batch K-Means
def cluster_passwords(embeddings, n_clusters=100):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, batch_size=1000)
    clusters = kmeans.fit_predict(embeddings)
    return clusters

# Save clusters to output directory
def save_clusters(clusters, output_dir):
    async_save_results(clusters, output_dir)

# Representative Password Selection
def select_representative_passwords(clusters, passwords, n_samples=10):
    representative_passwords = []
    for cluster in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster)[0]
        selected_indices = np.random.choice(cluster_indices, n_samples, replace=False)
        representative_passwords.extend([passwords[i] for i in selected_indices])
    return representative_passwords

# Run the entire preparation process
async def run_preparation(dataset_path, output="clusters", chunk_size=1000000):
    passwords = await async_load_passwords(dataset_path, chunk_size)
    model = PasswordEmbeddingModel().to("cuda")
    embeddings = extract_embeddings(model, passwords)
    clusters = cluster_passwords(embeddings)
    save_clusters(clusters, output)

# Example usage
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_preparation("large_password_dataset.txt"))
    representative_passwords = select_representative_passwords(clusters, passwords)
    print(representative_passwords)
