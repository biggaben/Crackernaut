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