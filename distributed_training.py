import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset

def setup_distributed(rank, world_size):
    """
    Setup distributed training environment
    
    Args:
        rank: Current process rank
        world_size: Total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleanup distributed training resources"""
    dist.destroy_process_group()

def train_distributed(rank, world_size, model, dataset, epochs=10, batch_size=32, 
                     learning_rate=0.001, save_path=None):
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
                                shuffle=True, drop_last=False)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )
    
    # Set up optimizer with learning rate scaling
    # Scale learning rate by number of processes (GPUs)
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate * world_size)
    
    # Training loop
    for epoch in range(epochs):
        # Set epoch for sampler
        sampler.set_epoch(epoch)
        
        # Train one epoch
        ddp_model.train()
        total_loss = 0.0
        
        for i, (inputs, targets) in enumerate(dataloader):
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
            
            if rank == 0 and i % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], "
                     f"Loss: {loss.item():.4f}")
        
        # Print epoch summary
        if rank == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {total_loss / len(dataloader):.4f}")
    
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
        model_class: Model class to instantiate
        dataset: Dataset for training
        num_gpus: Number of GPUs to use
        **kwargs: Additional arguments for train_distributed
    """
    import torch.multiprocessing as mp
    
    # Check available GPU count
    available_gpus = torch.cuda.device_count()
    if available_gpus < num_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available")
        num_gpus = available_gpus
    
    if num_gpus <= 1:
        print("Distributed training requires multiple GPUs. Falling back to single GPU training.")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model_class().to(device)
        # Call your regular training function here
        return
    
    # Launch processes
    mp.spawn(
        train_distributed,
        args=(num_gpus, model_class(), dataset) + tuple(kwargs.values()),
        nprocs=num_gpus,
        join=True
    )