import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('transformer_model')

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, embed_dim]
        return x + self.pe[:, :x.size(1)]


class PasswordTransformer(nn.Module):
    def __init__(self, vocab_size=128, embed_dim=64, num_heads=4, 
                 num_layers=3, hidden_dim=128, dropout=0.2, 
                 output_dim=6, max_length=20):
        """
        Lightweight transformer model for password scoring.
        
        Args:
            vocab_size: Size of vocabulary (ASCII characters)
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension in feed-forward layers
            dropout: Dropout rate
            output_dim: Number of output dimensions (variant types)
            max_length: Maximum password length
        """
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.max_length = max_length
        
    def forward(self, x, mask=None):
        """
        Forward pass through the transformer model.
        
        Args:
            x: Input tensor with password character indices
               Shape: [batch_size, seq_len]
            mask: Optional mask for padding
            
        Returns:
            Scores for each password variant type
        """
        # Create padding mask if none provided
        if mask is None and isinstance(x, torch.Tensor):
            mask = (x == 0)  # Assuming 0 is pad token
            
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        
        # Pass through transformer
        if mask is not None:
            x = self.transformer_encoder(x, src_key_padding_mask=mask)
        else:
            x = self.transformer_encoder(x)
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        
        # Final classification layer
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def save(self, path="ml_model.pth"):
        """Save the model to a file"""
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to {path}")
        
    def load(self, path="ml_model.pth", device=None):
        """Load model from a file"""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        if os.path.exists(path):
            self.load_state_dict(torch.load(path, map_location=device))
            self.to(device)
            self.eval()
            logger.info(f"Model loaded from {path}")
            return True
        logger.error(f"Model file not found: {path}")
        return False
    
    def encode_password(self, password, device=None):
        """
        Convert a password string to tensor representation
        
        Args:
            password: String password to encode
            device: Target device
            
        Returns:
            Tensor representation of the password
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Convert characters to indices (ASCII values)
        indices = [ord(c) for c in password[:self.max_length]]  # Truncate to max_length
        
        # Convert to tensor
        return torch.tensor(indices, dtype=torch.long, device=device).unsqueeze(0)
    
    def score_password(self, password, device=None):
        """
        Score a single password for different variant types
        
        Args:
            password: String password to score
            device: Target device
            
        Returns:
            Dictionary of scores for each variant type
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.eval()
        self.to(device)
        
        with torch.no_grad():
            encoded = self.encode_password(password, device)
            scores = self.forward(encoded)
            
        # Map output indices to variant types
        variant_types = ["Numeric", "Symbol", "Capitalization", "Leet", "Shift", "Repetition"]
        
        # Create a dictionary of scores
        return {variant_types[i]: scores[0][i].item() for i in range(min(len(variant_types), scores.size(1)))}
    
    def batch_score(self, passwords, device=None):
        """
        Score a batch of passwords
        
        Args:
            passwords: List of password strings
            device: Target device
            
        Returns:
            List of score dictionaries
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.eval()
        self.to(device)
        
        # Encode all passwords
        max_len = min(self.max_length, max(len(p) for p in passwords))
        batch = torch.zeros((len(passwords), max_len), dtype=torch.long, device=device)
        
        for i, password in enumerate(passwords):
            for j, char in enumerate(password[:max_len]):
                batch[i, j] = ord(char)
        
        # Create mask for padding
        mask = (batch == 0)
        
        # Get scores
        with torch.no_grad():
            scores = self.forward(batch, mask)
        
        # Map output indices to variant types
        variant_types = ["Numeric", "Symbol", "Capitalization", "Leet", "Shift", "Repetition"]
        
        # Create a list of dictionaries with scores
        results = []
        for i in range(len(passwords)):
            results.append({
                variant_types[j]: scores[i][j].item() 
                for j in range(min(len(variant_types), scores.size(1)))
            })
        
        return results