import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(PROJECT_ROOT, "config.json")

class SimplePasswordDataset(Dataset):
    """Simple dataset for password training data."""
    
    def __init__(self, data_path):
        """Initialize dataset from file path."""
        self.data = []
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = [line.strip() for line in f if line.strip()]
        else:
            # Fallback to dummy data
            self.data = ["password123", "admin", "123456", "qwerty"]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def setup_distributed(rank, world_size):
    """
    Setup distributed training environment
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    except Exception as e:
        print(f"Error setting up distributed training: {e}")
        raise

def cleanup_distributed():
    """Cleanup distributed training resources"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model, dataset, epochs=10, batch_size=32, 
                     learning_rate=0.001, weight_decay=1e-4, save_path=None):
    """
    Train model using distributed data parallelism
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
        model: Model to train
        dataset: Training dataset
        epochs: Number of epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        save_path: Path to save the model
    """
    # Setup distributed environment
    setup_distributed(rank, world_size)
    
    # Create model for this GPU
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create distributed sampler and dataloader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, 
                                shuffle=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Set up optimizer with learning rate scaling
    # Scale learning rate by number of processes (GPUs)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate * world_size, weight_decay=weight_decay)
    
    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Train one epoch
        ddp_model.train()
        total_loss = 0.0
        
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = ddp_model(inputs)
            loss = compute_loss(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print epoch summary
        if rank == 0:
            print(f"Epoch {epoch+1}: Avg Loss {total_loss/len(dataloader):.4f}")
    
    # Save model (only on rank 0)
    if rank == 0 and save_path:
        torch.save(model.state_dict(), save_path)
    
    # Cleanup
    cleanup_distributed()

def compute_loss(outputs, targets):
    """Compute loss based on your model's output format"""
    # Implement appropriate loss function
    # This is a placeholder
    criterion = nn.MSELoss()
    return criterion(outputs, targets)

def run_distributed_training(model_class, dataset, num_gpus, **kwargs):
    """
    Launch distributed training on multiple GPUs
    
    Args:
        model_class: Model class to instantiate        dataset: Dataset for training
        num_gpus: Number of GPUs to use
        **kwargs: Additional arguments for train_distributed
    """
    import multiprocessing as mp
    
    # Check available GPU count
    available_gpus = torch.cuda.device_count()
    num_gpus = min(num_gpus, available_gpus)
    
    if num_gpus <= 1:
        print("Distributed training requires multiple GPUs. Falling back to single GPU training.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_class().to(device)
        return train_single_gpu(model, dataset, **kwargs)
    
    # Launch processes
    mp.Process(
        target=train_distributed,
        args=(0, num_gpus, model_class(), dataset) + tuple(kwargs.values())
    ).start()

def train_single_gpu(model, dataset, epochs=10, batch_size=32, learning_rate=0.001, 
                    weight_decay=1e-4, save_path=None):
    """
    Train model using a single GPU or CPU when distributed training isn't available
    
    Args:
        model: Model to train
        dataset: Training dataset
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay parameter
        save_path: Path to save the model
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Create standard dataloader (no distributed sampler needed)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    # Set up optimizer (no learning rate scaling needed)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop (similar to distributed but simpler)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = compute_loss(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                     f"Loss: {loss.item():.4f}")
        
        # Print epoch summary
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(dataloader):.4f}")
    
    # Save model
    if save_path:
        torch.save(model.state_dict(), save_path)
    
    return model

def load_config(config_file):
    with open(config_file, 'r') as f:
        return json.load(f)

def load_dataset(dataset_path):
    try:
        return SimplePasswordDataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def main():
    config = load_config(CONFIG_FILE)
    model_class = globals().get(config["model_class"])
    if model_class is None:
        raise ValueError(f"Model class {config['model_class']} not found.")
    
    dataset = load_dataset(config["dataset_path"])

    run_distributed_training(
        model_class=model_class,
        dataset=dataset,
        num_gpus=config["num_gpus"],
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        save_path=config["save_path"]
    )

if __name__ == "__main__":
    main()