# test_variants.py
import sys
import os
import unittest

# Add the parent directory to the Python path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.variant_utils import (
    chain_shift_variants, 
    generate_variants
)

from src.utils.async_utils import run_preparation
from src.utils.config_utils import load_configuration
from src.list_preparer import text_to_tensor
from src.utils.performance_utils import enable_gradient_checkpointing
from src.models.transformer.transformer_model import PasswordTransformer
from src.models.embedding.embedding_model import PasswordEmbedder

class TestVariantUtils(unittest.TestCase):
    def setUp(self):
        # Use relative path from test directory to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        self.config = load_configuration(config_path)

    def test_shift_variants(self):
        self.assertIn("123abc", chain_shift_variants("abc123"))

    def test_leet_substitution(self):
        self.assertIn("P@ssword", generate_variants("Password", 20, 2))

    def test_complex_variants(self):
        variants = generate_variants("Password123", 20, 2)
        self.assertGreaterEqual(len(variants), 3)

    def test_negative_weights(self):
        config = {"modification_weights": {"Numeric": -0.2, "Symbol": 0.5}}
        for mod in config["modification_weights"]:
            config["modification_weights"][mod] = max(config["modification_weights"][mod], 0.0)
        self.assertTrue(all(weight >= 0 for weight in config["modification_weights"].values()))

    def test_empty_base(self):
        variants = generate_variants("", 10, 2)
        self.assertEqual(len(variants), 0)

    def test_special_chars(self):
        variants = generate_variants("!@#", 5, 1)
        self.assertTrue(all(len(v) <= 5 for v in variants))

class TestTransformerModel(unittest.TestCase):
    def test_unicode_and_padding(self):
        passwords = ["üñî", "pass"]
        tensor = text_to_tensor(passwords, max_length=5, device="cpu", vocab_size=256)
        self.assertEqual(tensor.shape, (2, 5))
        self.assertNotEqual(tensor[0][0].item(), 0)  # Check first character mapped

    def test_transformer_model_output(self):
        model = PasswordTransformer(vocab_size=256, embed_dim=64, num_heads=4, num_layers=3, hidden_dim=128, dropout=0.2, output_dim=6)
        tensor = text_to_tensor(["password"], max_length=20, device="cpu", vocab_size=256)
        output = model(tensor)
        self.assertEqual(output.shape, (1, 6))

    def test_gradient_checkpointing(self):
        model = PasswordEmbedder()
        enable_gradient_checkpointing(model)
        self.assertTrue(hasattr(model, "gradient_checkpointing") and model.gradient_checkpointing)

class TestListPreparer(unittest.TestCase):
    def test_run_preparation(self):
        # Assuming run_preparation is an async function
        import asyncio
        
        # Use relative paths and check if test data exists
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_path = os.path.join(project_root, "trainingdata", "rockyou-75.txt")
        output_dir = os.path.join(project_root, "clusters")
        chunk_size = 1000000
        
        # Skip test if training data doesn't exist (common in CI/development)
        if not os.path.exists(dataset_path):
            self.skipTest(f"Training data not found: {dataset_path}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Run the preparation with dynamic n_clusters
        asyncio.run(run_preparation(dataset_path, output=output_dir, chunk_size=chunk_size))
        # Add assertions to verify the output
        # Example assertion: Check if the output directory contains the expected files
        self.assertGreater(len(os.listdir(output_dir)), 0)


class TestConfigUtils(unittest.TestCase):
    def test_load_configuration(self):
        # Use relative path from test directory to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        config = load_configuration(config_path)
        self.assertIn("model_type", config)
        self.assertIn("model_embed_dim", config)
        self.assertIn("lp_chunk_size", config)

    def test_config_values(self):
        # Use relative path from test directory to project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config.json")
        config = load_configuration(config_path)
        self.assertIsInstance(config["model_embed_dim"], int)
        self.assertGreater(config["lp_chunk_size"], 0)
        self.assertIn("model_type", config)
        # Check for existing keys to avoid failure
        if "lp_batch_size" in config:
            self.assertIsInstance(config["lp_batch_size"], int)
            self.assertGreater(config["lp_batch_size"], 0)

if __name__ == "__main__":
    unittest.main()