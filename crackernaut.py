import random
import string
import tkinter as tk
from tkinter import filedialog
import hashlib
from typing import Generator, List
import logging
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
from itertools import islice

# GPU imports
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Initializes the CUDA driver automatically; used for side effects only.
from pycuda.compiler import SourceModule

# ---------------------- Logging Configuration ---------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------------------- Utility Functions ---------------------- #
def secure_hash(password: str) -> str:
    """Create a SHA256 hash of a given password."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()

def validate_input(text: str) -> bool:
    """Validate user input for basic security checks."""
    if not text or len(text) < 3:
        return False
    return all(c in string.printable for c in text)

def leetspeak(password: str) -> str:
    """Convert password to leetspeak."""
    replacements = {
        'a': '@', 'A': '@',
        'e': '3', 'E': '3',
        'i': '1', 'I': '1',
        'o': '0', 'O': '0',
        's': '$', 'S': '$'
    }
    return ''.join(replacements.get(c, c) for c in password)

# ---------------------- Advanced Human Branching ---------------------- #
def human_branch_variants(base: str, max_depth: int = 3) -> List[str]:
    """
    Generate variants of the base password using a recursive branching strategy,
    simulating human-like modifications.

    At each branch level, the following transformations are applied:
      - Append punctuation (e.g. "!", "?", "!?", "!?")
      - Substitute one letter at a time with a common leet equivalent
      - Append common digit sequences (e.g. "1", "123", "1234")
    """
    variants = set([base])
    current_set = set([base])
    
    punctuation_options = ["!", "?", "!?", "!?"]
    digit_options = ["1", "123", "1234"]
    leet_map = {'o': '0', 'a': '@', 'i': '1', 'e': '3', 's': '$'}

    for depth in range(max_depth):
        next_set = set()
        for cand in current_set:
            # 1. Append punctuation at the end
            for p in punctuation_options:
                next_set.add(cand + p)
            # 2. Apply one-character leet substitution (each substitution in a separate branch)
            for i, ch in enumerate(cand):
                lower = ch.lower()
                if lower in leet_map:
                    new_cand = cand[:i] + leet_map[lower] + cand[i+1:]
                    next_set.add(new_cand)
            # 3. Append digit sequences
            for d in digit_options:
                next_set.add(cand + d)
        variants.update(next_set)
        current_set = next_set
    return list(variants)

# ---------------------- Existing Candidate Generators ---------------------- #
def generate_base_variants(base_password: str) -> Generator[str, None, None]:
    """
    Generate basic variants using simple transformations.
    (These are useful in combination with dictionary-based variants.)
    """
    if not validate_input(base_password):
        logger.error("Invalid base password provided")
        return

    # Basic transformations
    yield base_password
    yield base_password[::-1]
    yield base_password.swapcase()
    yield leetspeak(base_password)

    # Common expansions and fixed punctuation variants
    common_expansions = ['!', '@', '#', '$', '123', '2023', '01', '001', '12', '1234']
    for exp in common_expansions:
        yield f"{base_password}{exp}"
        yield f"{exp}{base_password}"
    # Include one variant using a random punctuation from a fixed set
    common_punctuations = ["!", "@", "#", "$", "%", "&", "*", "?"]
    random_char = random.choice(common_punctuations)
    yield f"{base_password}{random_char}"
    yield f"{random_char}{base_password}"

def generate_candidates(dictionary_words: List[str], base_password: str) -> Generator[str, None, None]:
    """
    Generate candidate passwords by combining basic variants and dictionary words.
    """
    seen_passwords = set()

    def _add_unique(password: str) -> bool:
        if password not in seen_passwords:
            seen_passwords.add(password)
            return True
        return False

    # Yield basic variants
    for variant in generate_base_variants(base_password):
        if _add_unique(variant):
            yield variant

    # Combine with dictionary words
    for word in dictionary_words:
        if _add_unique(word):
            yield word
        combined = f"{word}{base_password}"
        if _add_unique(combined):
            yield combined
        combined = f"{base_password}{word}"
        if _add_unique(combined):
            yield combined

# ---------------------- Smart Filtering Functions ---------------------- #
def score_candidate(candidate: str, base: str) -> float:
    """
    Give a heuristic score to a candidate password.
    
    Scoring criteria (tweak as needed):
      +1 if candidate equals the base (case-insensitive)
      +1 if candidate is capitalized (first letter uppercase, rest lowercase)
      +1 if candidate uses common leet substitutions (contains '0', '@', '3', or '1')
      +1 if candidate starts or ends with a common punctuation character (!, @, #, $, %,&,*,?)
      -1 if candidate is exactly the reverse of the base
    """
    score = 0.0
    if candidate.lower() == base.lower():
        score += 1.0
    if candidate == candidate.capitalize():
        score += 1.0
    if any(sub in candidate for sub in ['0', '@', '3', '1']):
        score += 1.0
    common_punct = set("!@#$%&*?")
    if candidate[0] in common_punct or candidate[-1] in common_punct:
        score += 1.0
    if candidate == base[::-1]:
        score -= 1.0
    return score

def smart_filter_candidates(candidates: List[str], base: str, min_score: float = 0.5) -> List[str]:
    """
    Filter and sort candidates by their heuristic score.
    Only candidates with a score >= min_score are kept.
    """
    scored_candidates = [(c, score_candidate(c, base)) for c in candidates]
    filtered = [c for c, s in scored_candidates if s >= min_score]
    filtered.sort(key=lambda c: score_candidate(c, base), reverse=True)
    return filtered

# ---------------------- Hash Cracking Functions ---------------------- #
def crack_hash(hash_to_crack: str, candidates: Generator[str, None, None]) -> str:
    """
    Attempt to crack the hash using a candidate generator (CPU version).
    """
    for candidate in tqdm(candidates, desc="Checking candidates"):
        if secure_hash(candidate) == hash_to_crack:
            return candidate
    return None

def save_candidates(candidates: Generator[str, None, None], file_path: str) -> bool:
    """
    Save generated candidates to a file with progress tracking.
    """
    try:
        with open(file_path, 'w') as f:
            for candidate in tqdm(candidates, desc="Saving candidates"):
                f.write(f"{candidate}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving candidates: {e}")
        return False

def save_candidates_with_hash(candidates: List[str], file_path: str) -> bool:
    """
    Save generated candidates with their SHA256 hashes to a file.
    """
    try:
        with open(file_path, 'w') as f:
            for candidate in tqdm(candidates, desc="Saving candidates with hash"):
                candidate_hash = secure_hash(candidate)
                f.write(f"{candidate}  =>  {candidate_hash}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving candidates with hash: {e}")
        return False

# ---------------------- GPU Acceleration Code ---------------------- #
kernel_code = r"""
__device__ unsigned int rotr(unsigned int x, unsigned int n) {
    return (x >> n) | (x << (32 - n));
}
__device__ unsigned int sha256_ch(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ ((~x) & z);
}
__device__ unsigned int sha256_maj(unsigned int x, unsigned int y, unsigned int z) {
    return (x & y) ^ (x & z) ^ (y & z);
}
__device__ unsigned int sha256_bsigma0(unsigned int x) {
    return rotr(x,2) ^ rotr(x,13) ^ rotr(x,22);
}
__device__ unsigned int sha256_bsigma1(unsigned int x) {
    return rotr(x,6) ^ rotr(x,11) ^ rotr(x,25);
}
__device__ unsigned int sha256_ssigma0(unsigned int x) {
    return rotr(x,7) ^ rotr(x,18) ^ (x >> 3);
}
__device__ unsigned int sha256_ssigma1(unsigned int x) {
    return rotr(x,17) ^ rotr(x,19) ^ (x >> 10);
}
__global__ void sha256_kernel(const unsigned char *candidates, int num_candidates, int candidate_len, 
                                const unsigned int *target_hash, int *result_index) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_candidates) return;
    
    const unsigned char *candidate = candidates + idx * candidate_len;
    int len = 0;
    while (len < candidate_len && candidate[len] != 0) { len++; }
    
    unsigned char block[64];
    for (int i = 0; i < 64; i++) block[i] = 0;
    for (int i = 0; i < len; i++) { block[i] = candidate[i]; }
    if (len < 64) block[len] = 0x80;
    unsigned int bit_len = len * 8;
    block[63] = bit_len & 0xff;
    block[62] = (bit_len >> 8) & 0xff;
    block[61] = (bit_len >> 16) & 0xff;
    block[60] = (bit_len >> 24) & 0xff;
    
    unsigned int H0 = 0x6a09e667;
    unsigned int H1 = 0xbb67ae85;
    unsigned int H2 = 0x3c6ef372;
    unsigned int H3 = 0xa54ff53a;
    unsigned int H4 = 0x510e527f;
    unsigned int H5 = 0x9b05688c;
    unsigned int H6 = 0x1f83d9ab;
    unsigned int H7 = 0x5be0cd19;
    
    unsigned int W[64];
    for (int i = 0; i < 16; i++) {
        int j = i * 4;
        W[i] = ((unsigned int)block[j] << 24) | ((unsigned int)block[j+1] << 16) | 
               ((unsigned int)block[j+2] << 8) | ((unsigned int)block[j+3]);
    }
    for (int i = 16; i < 64; i++) {
        unsigned int s0 = sha256_ssigma0(W[i-15]);
        unsigned int s1 = sha256_ssigma1(W[i-2]);
        W[i] = W[i-16] + s0 + W[i-7] + s1;
    }
    
    const unsigned int K[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
        0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
        0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
        0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
        0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
        0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
    };
    
    for (int i = 0; i < 64; i++) {
        unsigned int T1 = H7 + sha256_bsigma1(H4) + sha256_ch(H4, H5, H6) + K[i] + W[i];
        unsigned int T2 = sha256_bsigma0(H0) + sha256_maj(H0, H1, H2);
        H7 = H6;
        H6 = H5;
        H5 = H4;
        H4 = H3 + T1;
        H3 = H2;
        H2 = H1;
        H1 = H0;
        H0 = T1 + T2;
    }
    
    if (H0 == target_hash[0] and H1 == target_hash[1] and
        H2 == target_hash[2] and H3 == target_hash[3] and
        H4 == target_hash[4] and H5 == target_hash[5] and
        H6 == target_hash[6] and H7 == target_hash[7]) {
        *result_index = idx;
    }
}
"""

def prepare_candidate_array(candidates: List[str], max_len: int) -> np.ndarray:
    """
    Convert list of candidate strings to a 2D NumPy array (uint8), padding/truncating to max_len.
    """
    arr = np.zeros((len(candidates), max_len), dtype=np.uint8)
    for i, cand in enumerate(candidates):
        encoded = cand.encode('ascii', errors='ignore')[:max_len]
        arr[i, :len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    return arr

def convert_target_hash(hash_hex: str) -> np.ndarray:
    """
    Convert a 64-character SHA256 hex string to an array of 8 uint32 values (big-endian).
    """
    bytes_hash = bytes.fromhex(hash_hex)
    target = np.array([int.from_bytes(bytes_hash[i*4:(i+1)*4], 'big') for i in range(8)], dtype=np.uint32)
    return target

def gpu_crack_hash(hash_to_crack: str, candidates_list: List[str]) -> str:
    """
    GPU-accelerated SHA256 hash cracking.
    """
    MAX_LEN = 55  # Maximum candidate length allowed (for one-block messages)
    candidates_array = prepare_candidate_array(candidates_list, MAX_LEN)
    num_candidates = candidates_array.shape[0]
    
    target_hash = convert_target_hash(hash_to_crack)
    
    candidates_gpu = cuda.mem_alloc(candidates_array.nbytes)
    cuda.memcpy_htod(candidates_gpu, candidates_array)
    
    target_hash_gpu = cuda.mem_alloc(target_hash.nbytes)
    cuda.memcpy_htod(target_hash_gpu, target_hash)
    
    result_index = np.array([-1], dtype=np.int32)
    result_index_gpu = cuda.mem_alloc(result_index.nbytes)
    cuda.memcpy_htod(result_index_gpu, result_index)
    
    mod = SourceModule(kernel_code)
    sha256_kernel = mod.get_function("sha256_kernel")
    
    block_size = 256
    grid_size = (num_candidates + block_size - 1) // block_size
    sha256_kernel(candidates_gpu, np.int32(num_candidates), np.int32(MAX_LEN),
                  target_hash_gpu, result_index_gpu,
                  block=(block_size, 1, 1), grid=(grid_size, 1))
    
    cuda.memcpy_dtoh(result_index, result_index_gpu)
    idx = result_index[0]
    if idx != -1:
        return candidates_list[idx]
    return None

# ---------------------- Mode Functions ---------------------- #
def mode_text_only(output_path: str = None, base_pass: str = None, combination_count: int = None):
    """Mode 1: Generate password combinations in text only."""
    logger.info("Mode 1: Password combinations (text only)")
    if base_pass is None:
        base_pass = input("Enter the base password guess: ").strip()
    if not validate_input(base_pass):
        logger.error("Invalid base password. Must be at least 3 characters with valid characters.")
        return
    common_words = [
        "password", "welcome", "company", "admin", "summer",
        "winter", "autumn", "spring", "qwerty", "abc123"
    ]
    # Generate basic dictionary variants
    dict_candidates = list(generate_candidates(common_words, base_pass))
    # Generate human-like branching variants
    human_candidates = human_branch_variants(base_pass, max_depth=3)
    # Combine both sets
    combined = set(dict_candidates + human_candidates)
    smart_candidates = smart_filter_candidates(list(combined), base_pass, min_score=0.5)
    if combination_count is not None:
        smart_candidates = smart_candidates[:combination_count]
    if output_path:
        if save_candidates(iter(smart_candidates), output_path):
            logger.info(f"Candidates successfully saved to: {output_path}")
        else:
            logger.error("Failed to save candidates.")
    else:
        logger.info("Generated (smart-filtered) password candidates:")
        for candidate in smart_candidates:
            print(candidate)

def mode_text_with_hash(output_path: str = None, base_pass: str = None, combination_count: int = None):
    """Mode 2: Generate password combinations with the password hash included."""
    logger.info("Mode 2: Password combinations with hash included")
    if base_pass is None:
        base_pass = input("Enter the base password guess: ").strip()
    if not validate_input(base_pass):
        logger.error("Invalid base password. Must be at least 3 characters with valid characters.")
        return
    common_words = [
        "password", "welcome", "company", "admin", "summer",
        "winter", "autumn", "spring", "qwerty", "abc123"
    ]
    dict_candidates = list(generate_candidates(common_words, base_pass))
    human_candidates = human_branch_variants(base_pass, max_depth=3)
    combined = set(dict_candidates + human_candidates)
    smart_candidates = smart_filter_candidates(list(combined), base_pass, min_score=0.5)
    if combination_count is not None:
        smart_candidates = smart_candidates[:combination_count]
    if output_path:
        if save_candidates_with_hash(smart_candidates, output_path):
            logger.info(f"Candidates with hashes successfully saved to: {output_path}")
        else:
            logger.error("Failed to save candidates with hashes.")
    else:
        logger.info("Generated (smart-filtered) password candidates with their SHA256 hashes:")
        for candidate in smart_candidates:
            candidate_hash = secure_hash(candidate)
            print(f"{candidate}  =>  {candidate_hash}")

def mode_recover(base_pass: str = None, combination_count: int = None):
    """Mode 3: Recover a password by checking common variants against its hash."""
    logger.info("Mode 3: Recover password")
    if base_pass is None:
        base_pass = input("Enter the base password guess: ").strip()
    if not validate_input(base_pass):
        logger.error("Invalid base password. Must be at least 3 characters with valid characters.")
        return
    hash_to_crack = input("Enter the SHA256 hash to attempt cracking: ").strip()
    if len(hash_to_crack) != 64:
        logger.error("Invalid SHA256 hash provided.")
        return
    common_words = [
        "password", "welcome", "company", "admin", "summer",
        "winter", "autumn", "spring", "qwerty", "abc123"
    ]
    dict_candidates = list(generate_candidates(common_words, base_pass))
    human_candidates = human_branch_variants(base_pass, max_depth=3)
    combined = set(dict_candidates + human_candidates)
    smart_candidates = smart_filter_candidates(list(combined), base_pass, min_score=0.5)
    if combination_count is not None:
        smart_candidates = smart_candidates[:combination_count]
    logger.info("Attempting to recover the password using GPU acceleration...")
    result = gpu_crack_hash(hash_to_crack, list(smart_candidates))
    if result:
        logger.info(f"SUCCESS: Found matching password: {result}")
    else:
        logger.info("No match found with the generated candidates.")

def mode_interactive(output_path: str = None, base_pass: str = None, combination_count: int = None):
    """Mode 4: Interactive menu for users unfamiliar with CLI."""
    logger.info("Interactive Mode")
    while True:
        print("\n--- Interactive Menu ---")
        print("1. Generate password combinations (text only)")
        print("2. Generate password combinations with hash included")
        print("3. Recover a password by checking common variants against its hash")
        print("4. Help menu")
        print("5. Exit")
        choice = input("Select an option (1-5): ").strip()
        if choice == "1":
            mode_text_only(output_path, base_pass, combination_count)
        elif choice == "2":
            mode_text_with_hash(output_path, base_pass, combination_count)
        elif choice == "3":
            mode_recover(base_pass, combination_count)
        elif choice == "4":
            mode_help()
        elif choice == "5":
            print("Exiting interactive mode.")
            break
        else:
            print("Invalid selection. Please try again.")

def mode_help():
    """Mode 5: Display help information."""
    help_text = """
Crackernaut 1.0 (GPU-Accelerated)
Author: BigGaben <crackernaut.extenuate388@passmail.net>
Homepage: https://github.com/biggaben/crackernaut

Usage: crackernaut.py [OPTIONS]

--help, -h               Print usage summary (Help menu)
--interactive, -i        Launch interactive mode, providing a user-friendly menu
--recover, -r            Recover a password by checking common variants
--hash, -H               Generate password combinations and print SHA256 hashes
-o, --output PATH        Define output file path to save generated candidate file(s)
-p, --password PASS      Specify base password guess (default: prompt for input)
-n, --number NUM         Number of combinations to generate (default: all)

Notes:
  • By default (no flags), the tool generates password variants from a given
    base password guess and prints them in plain text.
  • With --hash (-H), each generated candidate is accompanied by its SHA256 hash.
  • With --recover (-r), provide a SHA256 hash, and the tool attempts to crack it
    by generating variants and comparing their hashes (GPU-accelerated).
  • With --interactive (-i), you’ll enter a guided menu system without needing
    direct command-line knowledge.
  • With --output (-o), the generated file(s) will be saved to the specified path.
  • With --password (-p) and --number (-n), you can specify the base password and
    limit the number of combinations generated.
  • An advanced human-like branching algorithm is used to simulate realistic,
    intuitive password modifications (e.g. "fotboll!" → "fotboll!?" → "f0tb0ll!?" → etc.).

Examples:
  crackernaut.py
      Generates smart-filtered password variants (text only) from a prompt for the base password.

  crackernaut.py --hash -p fotboll -n 1000
      Uses "fotboll" as the base password and generates up to 1000 smart variants with their SHA256 hashes.

  crackernaut.py -r
      Prompts for a base password and a target hash, attempting to find a match.

  crackernaut.py --interactive
      Launches an interactive, menu-driven session.

  crackernaut.py -o /path/to/output.txt
      Saves the generated candidates to the specified file.

Disclaimer:
  Ensure you have explicit permission to use this tool when testing password
  security. Unauthorized use may violate local and federal laws. This tool is
  intended for educational and authorized penetration testing purposes only.
"""
    print(help_text)

# ---------------------- Main ---------------------- #
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--hash", "-H", action="store_true", help="Generate password combinations with hash (Mode 2)")
    parser.add_argument("--recover", "-r", action="store_true", help="Recover a password (Mode 3)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive mode (Mode 4)")
    parser.add_argument("--help", "-h", action="store_true", help="Show help menu (Mode 5)")
    parser.add_argument("-o", "--output", type=str, help="Define output file path to save generated candidate file(s)")
    parser.add_argument("-p", "--password", type=str, help="Specify base password guess")
    parser.add_argument("-n", "--number", type=int, help="Number of combinations to generate")
    args, unknown = parser.parse_known_args()
    
    base_pass = args.password
    combination_count = args.number
    
    if args.help:
        mode_help()
    elif args.interactive:
        mode_interactive(output_path=args.output, base_pass=base_pass, combination_count=combination_count)
    elif args.recover:
        mode_recover(base_pass, combination_count)
    elif args.hash:
        mode_text_with_hash(output_path=args.output, base_pass=base_pass, combination_count=combination_count)
    else:
        mode_text_only(output_path=args.output, base_pass=base_pass, combination_count=combination_count)

if __name__ == "__main__":
    main()
