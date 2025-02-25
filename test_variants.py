# test_variants.py
from variant_utils import (
    chain_shift_variants, 
    generate_variants,
    generate_variants
)

def test_shift_variants():
    assert "123abc" in chain_shift_variants("abc123")

def test_leet_substitution():
    assert "P@ssword" in generate_variants("Password")

def test_complex_variants():
    variants = generate_variants("Password123", 20, 2)
    assert len(variants) >= 3  # Ensure minimum variant generation
    
def test_negative_weights():
    config = {"modification_weights": {"Numeric": -0.2, "Symbol": 0.5}}
    from config_utils import load_configuration
    config = load_configuration()  # Simulate loading
    assert all(weight >= 0 for weight in config["modification_weights"].values())

def test_empty_base():
    variants = generate_variants("", 10, 2)
    assert len(variants) == 0

def test_special_chars():
    variants = generate_variants("!@#", 5, 1)
    assert all(len(v) <= 5 for v in variants)

import unittest
import torch
from list_preparer import text_to_tensor, PasswordEmbeddingModel
from performance_utils import enable_gradient_checkpointing
from transformer_model import PasswordTransformer
from variant_utils import generate_variants

class TestCrackernaut(unittest.TestCase):
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
        model = PasswordEmbeddingModel()
        enable_gradient_checkpointing(model)
        self.assertTrue(hasattr(model, "gradient_checkpointing") and model.gradient_checkpointing)

if __name__ == "__main__":
    unittest.main()