import asyncio
import aiofiles
import torch
import numpy as np
import os

from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
from pathlib import Path

# Step 1: Load passwords from a file with progress
async def load_password_chunks(file_path: str, chunk_size=1000000, total_lines=None):
    """Load passwords in chunks with deduplication and progress tracking."""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        chunk = []
        seen = set()
        # Initialize tqdm with total if known, else indeterminate
        pbar = tqdm(total=total_lines, desc="Loading passwords", unit="lines") if total_lines else tqdm(desc="Loading passwords", unit="lines")
        async for line in f:
            pwd = line.strip()
            if len(pwd) >= 4 and pwd not in seen:  # Basic filtering
                chunk.append(pwd)
                seen.add(pwd)
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
                seen.clear()
                pbar.update(chunk_size)
        if chunk:
            yield chunk
            pbar.update(len(chunk))
        pbar.close()

# Step 2: Convert text to tensor for model input
def text_to_tensor(passwords, max_length=20, device="cuda", vocab_size=128):
    """Convert passwords to a padded tensor."""
    batch = []
    for pwd in passwords:
        indices = [ord(c) % vocab_size for c in pwd[:max_length]]
        indices += [0] * (max_length - len(indices)) if len(indices) < max_length else []
        batch.append(indices)
    return torch.tensor(batch, dtype=torch.long, device=device)

# Step 3: Extract embeddings with progress
def extract_embeddings(model, password_list, batch_size=1024, device="cuda", max_length=20):
    """Extract embeddings in batches with progress tracking."""
    model.eval()
    embeddings = []
    # Progress bar for embedding extraction
    pbar = tqdm(total=len(password_list), desc="Extracting embeddings", unit="passwords")
    for i in range(0, len(password_list), batch_size):
        batch = password_list[i:i + batch_size]
        tensors = text_to_tensor(batch, max_length=max_length, device=device)
        with torch.no_grad():
            emb = model(tensors)
        embeddings.append(emb.cpu().numpy())
        pbar.update(len(batch))
    pbar.close()
    return np.vstack(embeddings)

# Step 4: Cluster embeddings
def cluster_passwords(embeddings, n_clusters=1000):
    """Cluster embeddings using MiniBatchKMeans."""
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_

# Step 5: Save clusters with progress
def save_clusters(passwords, labels, embeddings, output_dir="clusters"):
    """Save cluster representatives with progress tracking."""
    Path(output_dir).mkdir(exist_ok=True)
    clusters = {}
    for pwd, label, emb in zip(passwords, labels, embeddings):
        clusters.setdefault(label, []).append((pwd, emb))
    
    # Select representative password per cluster
    reps = {}
    for label, items in clusters.items():
        passwords, embs = zip(*items)
        centroid = np.mean(embs, axis=0)
        rep_idx = np.argmin([np.linalg.norm(e - centroid) for e in embs])
        reps[label] = passwords[rep_idx]
    
    # Progress bar for saving
    pbar = tqdm(total=len(reps), desc="Saving clusters", unit="clusters")
    for label, rep in reps.items():
        cluster_file = os.path.join(output_dir, f"cluster_{label}.txt")
        with open(cluster_file, "w", encoding="utf-8") as f:
            f.write(rep)
        pbar.update(1)
    pbar.close()

# Main function to run the pipeline
async def prepare_list(dataset_path: str, output_dir="clusters"):
    """Run the list preparation pipeline with progress bars."""
    # Count lines for accurate progress (optional, can skip for simplicity)
    total_lines = sum(1 for _ in open(dataset_path, 'r', encoding='utf-8'))
    print(f"Dataset has {total_lines} lines")

    # Load passwords
    all_passwords = []
    async for chunk in load_password_chunks(dataset_path, chunk_size=1000000, total_lines=total_lines):
        all_passwords.extend(chunk)
    print(f"Loaded {len(all_passwords)} unique passwords")

    # Initialize model and device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PasswordTransformer().to(device)

    # Extract embeddings
    embeddings = extract_embeddings(model, all_passwords, device=device)

    # Cluster passwords
    n_clusters = max(1, len(all_passwords) // 10)  # Heuristic: 1 cluster per 10 passwords
    labels, _ = cluster_passwords(embeddings, n_clusters=n_clusters)

    # Save results
    save_clusters(all_passwords, labels, embeddings, output_dir=output_dir)
    print(f"Preparation complete. Clusters saved in {output_dir}")

# Run the pipeline
if __name__ == "__main__":
    asyncio.run(prepare_list("passwords.txt"))