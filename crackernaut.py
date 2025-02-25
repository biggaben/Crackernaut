#!/usr/bin/env python3
"""
crackernaut.py

Main production script for Crackernaut – a utility that generates human-like
password variants from a given base password. The script loads configuration
parameters (from a JSON file or defaults), generates variants via transformation
chains, applies smart filtering based on a machine learning model, and outputs
the top variants to the console or a specified file.

Usage:
    python crackernaut.py -p <base_password> [-l <max_length>] [-n <output_count>] [-o <output_file>]
    
    If -p is not provided, the script prompts for the base password.
    The -l flag sets the maximum length of variants.
    The -n flag limits the number of variants output.
    The -o flag saves the output variants to a file.
"""

import argparse
import os
import re
import sys
import torch
from config_utils import load_configuration
from variant_utils import generate_variants, SYMBOLS
from cuda_ml import load_ml_model, extract_features, predict_config_adjustment
from tqdm import tqdm
import queue
import threading
from variant_utils import generate_variants_parallel

class VariantPipeline:
    """
    Pipeline for concurrent variant generation and scoring.
    Uses producer-consumer pattern to maximize resource utilization.
    """
    
    def __init__(self, model, config, base_password, max_length=20, chain_depth=3, 
                 queue_size=1000, num_producers=1, num_consumers=None):
        """
        Initialize the variant processing pipeline.
        
        Args:
            model: ML model for scoring variants
            config: Configuration parameters
            base_password: Base password to generate variants from
            max_length: Maximum length of variants
            chain_depth: Chain depth for variant generation
            queue_size: Size of internal queues
            num_producers: Number of producer threads
            num_consumers: Number of consumer threads (defaults to CPU count)
        """
        self.model = model
        self.config = config
        self.base = base_password
        self.max_length = max_length
        self.chain_depth = chain_depth
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        
        # Set up queues
        self.variant_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        
        # Set up workers
        self.num_producers = num_producers
        self.num_consumers = num_consumers if num_consumers else max(1, os.cpu_count() - 1)
        
        # Status flags
        self.stop_requested = False
        self.producers_done = 0
        self.consumers_done = 0
        
        # Configure weights for scoring
        self.weights = torch.tensor(self.config['weights'], dtype=torch.float32, device=self.device).view(-1, 1)
    
    def producer_task(self):
        """Generate variants and add them to the queue"""
        try:
            # Generate variants in batches to avoid overwhelming the queue
            variant_batches = []
            batch_size = 1000
            all_variants = generate_variants_parallel(self.base, self.max_length, self.chain_depth)
            
            # Split into batches
            for i in range(0, len(all_variants), batch_size):
                variant_batches.append(all_variants[i:i+batch_size])
                
            for batch in variant_batches:
                if self.stop_requested:
                    break
                # Feed variants into queue
                for variant in batch:
                    if self.stop_requested:
                        break
                    self.variant_queue.put(variant)
                
        except Exception as e:
            print(f"Producer error: {e}")
        finally:
            with threading.Lock():
                self.producers_done += 1
                if self.producers_done == self.num_producers:
                    # Signal that all variants have been produced
                    for _ in range(self.num_consumers):
                        self.variant_queue.put(None)
    
    def consumer_task(self, consumer_id):
        """Score variants from the queue"""
        try:
            while not self.stop_requested:
                # Get next variant
                variant = self.variant_queue.get()
                if variant is None:
                    # End signal received
                    break
                
                try:
                    # Process variant
                    with torch.no_grad():
                        features = extract_features(variant, self.base, self.config['feature_config'])
                        features_tensor = features.to(self.device)
                        score = float((self.model(features_tensor.unsqueeze(0)) @ self.weights).item())
                    
                    # Store result
                    self.result_queue.put((variant, score))
                except Exception as e:
                    print(f"Error processing variant {variant}: {e}")
                finally:
                    self.variant_queue.task_done()
        
        except Exception as e:
            print(f"Consumer {consumer_id} error: {e}")
        finally:
            with threading.Lock():
                self.consumers_done += 1
                if self.consumers_done == self.num_consumers:
                    # Signal that all scoring is done
                    self.result_queue.put(None)
    
    def run(self):
        """Run the pipeline and return scored variants"""
        # Start threads
        producer_threads = [threading.Thread(target=self.producer_task) 
                           for _ in range(self.num_producers)]
        
        consumer_threads = [threading.Thread(target=self.consumer_task, args=(i,)) 
                           for i in range(self.num_consumers)]
        
        # Start all threads
        for thread in producer_threads + consumer_threads:
            thread.daemon = True
            thread.start()
        
        # Collect results
        results = []
        progress_bar = None
        try:
            while self.consumers_done < self.num_consumers:
                result = self.result_queue.get(timeout=1.0)
                if result is None:
                    break
                results.append(result)
                
                # Update progress bar
                if progress_bar is None:
                    progress_bar = tqdm(desc="Processing variants")
                progress_bar.update(1)
        except queue.Empty:
            # Just retry if queue is temporarily empty
            pass
        finally:
            if progress_bar:
                progress_bar.close()
        
        # Wait for threads to finish
        for thread in producer_threads + consumer_threads:
            thread.join(timeout=0.5)
        
        # Sort by score
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def stop(self):
        """Stop the pipeline gracefully"""
        self.stop_requested = True

# Helper function to maintain compatibility with existing code
def pipeline_variant_processing(base, max_length, chain_depth, model, config):
    """Process variants using pipeline parallelism"""
    pipeline = VariantPipeline(model, config, base, max_length, chain_depth)
    return pipeline.run()

def score_variants_optimized(variants, base, model, config, device=None):
    """
    Score password variants with optimized batch processing using GPU when available.
    
    Args:
        variants: List of password variants to score
        base: Base password for reference
        model: Trained scoring model
        config: Configuration parameters
        device: Computation device (auto-detected if None)
        
    Returns:
        List of (variant, score) tuples sorted by score
    """
    if not variants:
        return []
        
    # Determine device if not specified
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Calculate optimal batch size based on available memory
    if torch.cuda.is_available():
        # Get free memory and estimate conservatively (in bytes)
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        # Estimate 1KB per variant feature vector for safety
        optimal_batch_size = min(len(variants), max(32, int(free_memory * 0.7 / 1024)))
    else:
        optimal_batch_size = 64  # Default CPU batch size
    
    model = model.to(device)
    model.eval()
    
    # Get config weights
    weights = torch.tensor(config['weights'], dtype=torch.float32, device=device).view(-1, 1)
    
    # Process variants in batches
    scored_variants = []
    with torch.no_grad():
        for i in tqdm(range(0, len(variants), optimal_batch_size), desc="Scoring variants"):
            batch = variants[i:i+optimal_batch_size]
            
            # Extract features for entire batch
            batch_features = []
            for variant in batch:
                features = extract_features(variant, base, config['feature_config'])
                batch_features.append(features)
            
            # Convert to tensor and process
            features_batch = torch.stack(batch_features).to(device)
            scores = model(features_batch) @ weights
            
            # Add to results
            for j, (variant, score) in enumerate(zip(batch, scores.cpu().numpy().flatten())):
                scored_variants.append((variant, float(score)))
    
    # Sort by score (descending)
    return sorted(scored_variants, key=lambda x: x[1], reverse=True)

def process_large_wordlist(wordlist_path, model, config, batch_size=1000, max_passwords=None):
    """
    Process a large wordlist in memory-efficient batches.
    
    Args:
        wordlist_path: Path to the wordlist file
        model: ML model for processing
        config: Configuration parameters
        batch_size: Number of passwords to process at once
        max_passwords: Maximum number of passwords to process (None = all)
        
    Returns:
        List of processed results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    results = []
    count = 0
    
    try:
        # Count lines for progress bar
        if max_passwords is None:
            with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_length = sum(1 for _ in f)
            total = file_length
        else:
            total = max_passwords
            
        # Process in batches
        with tqdm(total=total, desc="Processing wordlist") as pbar:
            with open(wordlist_path, 'r', encoding='utf-8', errors='ignore') as f:
                batch = []
                
                for line in f:
                    if max_passwords is not None and count >= max_passwords:
                        break
                        
                    password = line.strip()
                    if not password:
                        continue
                        
                    batch.append(password)
                    count += 1
                    
                    if len(batch) >= batch_size:
                        batch_results = process_password_batch(batch, model, config, device)
                        results.extend(batch_results)
                        pbar.update(len(batch))
                        batch = []
                
                # Process remaining passwords
                if batch:
                    batch_results = process_password_batch(batch, model, config, device)
                    results.extend(batch_results)
                    pbar.update(len(batch))
                    
        return results
    
    except Exception as e:
        print(f"Error processing wordlist: {e}")
        # Return partial results if any
        return results

def process_password_batch(passwords, model, config, device):
    """Process a batch of passwords using the model"""
    results = []
    
    with torch.no_grad():
        # Extract features for all passwords in batch
        features_list = []
        for password in passwords:
            features = extract_features_for_wordlist(password, config)
            features_list.append(features)
        
        # Process features in one go
        features_batch = torch.stack(features_list).to(device)
        outputs = model(features_batch)
        
        # Process outputs
        for i, (password, output) in enumerate(zip(passwords, outputs)):
            # Process according to your model's output format
            result = process_model_output(password, output, config)
            results.append(result)
    
    return results

# Helper functions (implement according to your existing code)
def extract_features_for_wordlist(password, config):
    """Extract features from password for wordlist processing"""
    # Implement feature extraction logic based on your existing code
    # This is a placeholder
    return torch.zeros(config['feature_size'])

def process_model_output(password, output, config):
    """Process model output for a password"""
    # Implement output processing logic
    # This is a placeholder
    score = float(output.sum().item())
    return (password, score)

def main():
    """Main function to generate and output password variants."""
    parser = argparse.ArgumentParser(description="Crackernaut Variant Generator")
    parser.add_argument("-p", "--password", type=str, help="Base password (if not provided, prompt for input)")
    parser.add_argument("-l", "--length", type=int, help="Maximum length of generated variants")
    parser.add_argument("-n", "--number", type=int, help="Limit number of variants to output")
    parser.add_argument("-o", "--output", type=str, help="File to save the variants")
    args = parser.parse_args()
    
    # Load configuration settings from config_utils
    config = load_configuration()
    
    # Load the ML model from cuda_ml and determine the device (CPU or CUDA)
    model = load_ml_model()
    device = next(model.parameters()).device
    
    if args.password:
        base = args.password.strip()
    else:
        base = input("Enter base password: ").strip()
    
    # Validate the base password: must contain letters and be at least 3 characters
    while not base or len(base) < 3 or not re.search(r"[a-zA-Z]", base):
        print("Invalid base password. Must contain letters and be at least 3 characters.")
        base = input("Enter base password: ").strip()
    
    # Override the config's max_length if provided via args
    if args.length:
        config["max_length"] = args.length
    
    # Generate variants using variant_utils
    variants = generate_variants(base, config["max_length"], config["chain_depth"])
    
    if not variants:
        print("No variants generated. Try adjusting the configuration.")
        return
    
    # Add progress indication for variant generation
    print(f"Scoring {len(variants)} variants...")
    scored_variants = score_variants_optimized(variants, base, model, config, device)
    
    if not scored_variants:
        print("No variants meet the scoring criteria.")
        return
    
    # Select the top N variants if specified, otherwise select all
    if args.number:
        output_variants = [var for var, score in scored_variants[:args.number]]
    else:
        output_variants = [var for var, score in scored_variants]
    
    # Handle output: save to file if specified, otherwise print to console
    if args.output:
        try:
            # Check if the output file already exists
            if os.path.exists(args.output):
                choice = input(f"File {args.output} exists. Overwrite (o), append (a), or cancel (c)? ").strip().lower()
                if choice == 'o':
                    mode = 'w'  # Overwrite
                elif choice == 'a':
                    mode = 'a'  # Append
                else:
                    print("Operation canceled.")
                    return
            else:
                mode = 'w'  # Write new file
            # Write variants to the file
            with open(args.output, mode) as f:
                for variant in output_variants:
                    f.write(variant + "\n")
            print(f"Variants saved to {args.output}")
        except IOError as e:
            print(f"Error saving variants: {e}")
    else:
        # Output variants to the console
        print("Generated Variants:")
        for variant in output_variants:
            print(variant)

if __name__ == "__main__":
    main()