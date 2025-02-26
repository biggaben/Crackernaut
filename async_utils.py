# async_utils.py
import asyncio
import aiofiles
import os
import logging
from typing import List, Optional, Set, Dict, Any

async def async_load_passwords(
    wordlist_path: str, 
    max_count: Optional[int] = None,
    min_length: int = 0, 
    max_length: int = 100,
    skip_lines: int = 0
) -> List[str]:
    """
    Asynchronously load passwords from file with filtering options.
    
    Args:
        wordlist_path: Path to the wordlist file
        max_count: Maximum number of passwords to load (None = all)
        min_length: Minimum password length
        max_length: Maximum password length
        skip_lines: Number of lines to skip from the beginning
        
    Returns:
        List of loaded passwords
    """
    passwords = []
    
    try:
        async with aiofiles.open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            skipped = 0
            count = 0
            
            # Skip lines if needed
            async for _ in f:
                skipped += 1
                if skipped >= skip_lines:
                    break
            
            # Read passwords
            async for line in f:
                if max_count is not None and count >= max_count:
                    break
                
                password = line.strip()
                if not password:
                    continue
                
                if min_length <= len(password) <= max_length:
                    passwords.append(password)
                    count += 1
        
        return passwords
    except Exception as e:
        logging.error(f"Error loading wordlist: {e}")
        return []

async def async_save_results(
    results: List[tuple], 
    output_path: str, 
    header: Optional[str] = None,
    append: bool = False
) -> bool:
    """
    Asynchronously save results to file.
    
    Args:
        results: List of result tuples to save
        output_path: Output file path
        header: Optional header string
        append: Whether to append to existing file
        
    Returns:
        Success status
    """
    mode = 'a' if append else 'w'
    try:
        async with aiofiles.open(output_path, mode, encoding='utf-8') as f:
            if header and not append:
                await f.write(f"{header}\n")
            
            for result in results:
                if isinstance(result, tuple) and len(result) >= 2:
                    password, score = result[0], result[1]
                    await f.write(f"{password},{score}\n")
                else:
                    await f.write(f"{result}\n")
        
        return True
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        return False

async def async_process_directory(
    directory_path: str,
    pattern: str = "*.txt",
    processor_func: callable = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Process multiple files in a directory asynchronously.
    
    Args:
        directory_path: Directory containing files
        pattern: File pattern to match
        processor_func: Function to process each file
        **kwargs: Additional arguments for processor_func
        
    Returns:
        Dictionary of results by filename
    """
    import glob
    
    results = {}
    files = glob.glob(os.path.join(directory_path, pattern))
    
    # Create tasks for each file
    tasks = []
    for file_path in files:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            if processor_func:
                task = asyncio.create_task(processor_func(file_path, **kwargs))
                tasks.append((filename, task))
    
    # Await all tasks
    for filename, task in tasks:
        try:
            result = await task
            results[filename] = result
        except Exception as e:
            logging.error(f"Error processing {filename}: {e}")
            results[filename] = None
    
    return results

async def run_preparation(dataset_path: str, output: str, chunk_size: int = 10000) -> None:
    """
    Asynchronously prepare a dataset by splitting it into manageable chunks.
    
    Args:
        dataset_path: Path to the password dataset file
        output: Directory to store output chunks
        chunk_size: Number of passwords per chunk
    """
    # Ensure output directory exists
    os.makedirs(output, exist_ok=True)
    
    logging.info(f"Loading passwords from {dataset_path}")
    
    # Load passwords from dataset
    passwords = await async_load_passwords(dataset_path)
    
    if not passwords:
        logging.error(f"No passwords loaded from {dataset_path}")
        return
    
    total_passwords = len(passwords)
    logging.info(f"Loaded {total_passwords} passwords, splitting into chunks of {chunk_size}")
    
    # Calculate number of chunks
    total_chunks = (total_passwords + chunk_size - 1) // chunk_size
    
    # Process and save each chunk
    for i in range(total_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total_passwords)
        chunk = passwords[start_idx:end_idx]
        
        # Format output filename with leading zeros for proper sorting
        chunk_file = os.path.join(output, f"chunk_{i+1:04d}.txt")
        
        # Save chunk with empty scores (just the passwords)
        await async_save_results([(pw, "") for pw in chunk], chunk_file)
        
        logging.info(f"Saved chunk {i+1}/{total_chunks} ({len(chunk)} passwords) to {chunk_file}")
    
    logging.info(f"Dataset preparation complete: {total_chunks} chunks saved to {output}")

from typing import List

def load_passwords(wordlist_path: str, max_count: int = 100000) -> List[str]:
    """Load passwords from a wordlist file with limit."""
    try:
        with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f.readlines()[:max_count] if line.strip()]
    except Exception as e:
        print(f"Error loading wordlist: {e}")
        return []