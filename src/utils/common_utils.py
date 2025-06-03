import re
import string
import torch
import torch.nn as nn

def extract_features(variant: str, base: str, device: str = "cpu") -> torch.Tensor:
	"""
	Extracts an 8D feature vector from a (base, variant) pair on the specified device.
	"""
	def numeric_suffix_len(s: str) -> int:
		match = re.search(r"(\d+)$", s)
		return len(match.group(1)) if match else 0

	f1 = abs(numeric_suffix_len(variant) - numeric_suffix_len(base))
	f2 = sum(ch in string.punctuation for ch in variant)

	min_len = min(len(base), len(variant))
	cap_diff = sum(1 for i in range(min_len) if base[i].islower() != variant[i].islower())
	f3 = float(cap_diff)

	leet_chars = {"0", "3", "1", "$"}
	f4 = sum(ch in leet_chars for ch in variant)

	m_b = re.search(r"\d+$", base)
	m_v = re.search(r"^\d+", variant)
	f5 = 1 if m_b and m_v and m_b.group() == m_v.group() else 0

	f6 = abs(len(variant) - len(base))

	symbol_positions = [i for i, c in enumerate(variant) if c in string.punctuation]
	f7 = len(symbol_positions) / len(variant) if variant else 0

	total_leet = sum(1 for c in variant if c in leet_chars)
	f8 = total_leet / len(variant) if variant else 0

	features = [f1, f2, f3, f4, f5, f6, f7, f8]
	return torch.tensor(features, dtype=torch.float32, device=device).unsqueeze(0)

class MLP(nn.Module):
	def __init__(self, input_dim=8, hidden_dim=64, output_dim=6, dropout=0.2):
		super().__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, hidden_dim)
		self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
		self.fc4 = nn.Linear(hidden_dim // 2, output_dim)
		self.dropout = nn.Dropout(dropout)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.relu(self.fc1(x))
		x = self.dropout(x)
		x = self.relu(self.fc2(x))
		x = self.dropout(x)
		x = self.relu(self.fc3(x))
		x = self.dropout(x)
		x = self.fc4(x)
		return x