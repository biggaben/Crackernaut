# performance_utils.py
import os
import torch
import logging
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from models.embedding.embedding_model import PasswordEmbedder

def get_optimal_workers() -> int:
    """Get optimal number of worker processes based on system resources"""
    cpu_count = os.cpu_count() or 4
    return max(1, cpu_count - 1)  # Leave one core for system tasks

def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices"""
    info = {
        'cpu_count': os.cpu_count(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_devices': 0,
        'cuda_memory': [],
        'optimal_workers': get_optimal_workers()
    }
    
    if info['cuda_available']:
        info['cuda_devices'] = torch.cuda.device_count()
        for i in range(info['cuda_devices']):
            props = torch.cuda.get_device_properties(i)
            info['cuda_memory'].append({
                'device': i,
                'name': props.name,
                'total_memory_gb': round(props.total_memory / (1024**3), 2),
                'free_memory_gb': round((props.total_memory - torch.cuda.memory_allocated(i)) / (1024**3), 2)
            })
    
    return info

def optimize_batch_size(
    model_size_mb: float, 
    feature_vector_size: int,
    device_id: int = 0,
    min_batch: int = 16,
    max_batch: int = 1024,
    memory_factor: float = 0.7
) -> int:
    """
    Calculate optimal batch size based on available GPU memory.
    
    Args:
        model_size_mb: Approximate model size in MB
        feature_vector_size: Size of feature vectors
        device_id: CUDA device ID
        min_batch: Minimum batch size
        max_batch: Maximum batch size
        memory_factor: Memory utilization factor (0.0-1.0)
        
    Returns:
        Optimal batch size
    """
    # Add: "For transformer models, adjust feature_vector_size based on the model's embed_dim."
    if not torch.cuda.is_available():
        return min_batch
    
    try:
        # Calculate available memory in bytes
        free_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        available_memory = free_memory - allocated_memory
        
        # Calculate memory for model (convert MB to bytes)
        model_memory = model_size_mb * 1024 * 1024
        
        # Calculate memory per sample (32-bit float = 4 bytes)
        bytes_per_sample = feature_vector_size * 4
        
        # Calculate batch size with safety margin
        usable_memory = (available_memory - model_memory) * memory_factor
        optimal_batch = int(usable_memory / bytes_per_sample)
        
        # Clamp to reasonable range
        return max(min_batch, min(max_batch, optimal_batch))
    
    except Exception as e:
        logging.warning(f"Error calculating optimal batch size: {e}")
        return min_batch

def choose_parallelism_strategy(task_type: str, device_info: Dict[str, Any]) -> str:
    """
    Determine best parallelism strategy based on task type and available resources.
    
    Args:
        task_type: Type of task ('training', 'inference', 'generation')
        device_info: Device information from get_device_info()
        
    Returns:
        Recommended strategy name
    """
    if task_type == 'training':
        if device_info['cuda_available'] and device_info['cuda_devices'] > 1:
            return 'distributed_data_parallel'
        elif device_info['cuda_available']:
            return 'single_gpu'
        else:
            return 'data_parallel'
    
    elif task_type == 'inference':
        if device_info['cuda_available']:
            return 'batched_gpu'
        else:
            return 'process_pool'
    
    elif task_type == 'generation':
        # Password generation is CPU-bound, use process pool
        return 'process_pool'
    
    return 'sequential'  # Default fallback

def enable_gradient_checkpointing(model):
    if isinstance(model, PasswordEmbedder):
        model.gradient_checkpointing = True
        print("Gradient checkpointing enabled.")
    else:
        print("Gradient checkpointing not supported for this model.")