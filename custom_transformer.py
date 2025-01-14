#!/usr/bin/env python3
"""
Custom Transformer Model
"""
# Standard Library Imports
from dataclasses import dataclass
import math

# Third Party Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Local Imports
from custom_layers import linear, gelu, layer_norm, attention

@dataclass
class TransformerConfig:
    """
    Configuration for the Transformer model
    """
    vocab_size: int = 50257
    max_seq_len: int = 2048
    dim: int = 768
    num_layers: int = 2
    num_heads: int = 2
    dropout: float = 0.1


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.dim = config.dim
        self.head_dim = config.dim // config.num_heads
        
        self.q_proj = nn.Parameter(torch.randn(config.dim, config.dim))
        self.k_proj = nn.Parameter(torch.randn(config.dim, config.dim))
        self.v_proj = nn.Parameter(torch.randn(config.dim, config.dim))
        self.out_proj = nn.Parameter(torch.randn(config.dim, config.dim))
    
    def forward(self, x, use_cache=False):
        """
        Forward pass for the multi-head attention mechanism
        """
        batch_size, seq_len, _ = x.shape
        
        q = linear(x, self.q_proj)
        k = linear(x, self.k_proj)
        v = linear(x, self.v_proj)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Create causal mask and expand it for batch size and heads
        mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        mask = mask.expand(batch_size, self.num_heads, seq_len, seq_len)
        mask = mask.to(x.device)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch_size, num_heads, seq_len, seq_len]
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention weights
        attn_output = torch.matmul(attn_weights, v)  # [batch_size, num_heads, seq_len, head_dim]
        
        # Reshape back to [batch_size, seq_len, dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        
        # Final projection
        output = linear(attn_output, self.out_proj)
        return output


class LayerNormModule(nn.Module):
    """
    Layer normalization module for the Transformer model
    """
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        """
        Forward pass for the layer normalization
        """
        normalized = layer_norm(x)
        return normalized * self.weight.view(1, 1, -1) + self.bias.view(1, 1, -1)


class FeedForward(nn.Module):
    """
    Feed-forward network module for the Transformer model
    """
    def __init__(self, config):
        super().__init__()
        # Initialize weights with correct dimensions
        self.w1 = nn.Parameter(torch.randn(config.dim, 4 * config.dim))  # [dim, 4*dim]
        self.w2 = nn.Parameter(torch.randn(4 * config.dim, config.dim))  # [4*dim, dim]
    
    def forward(self, x):
        """
        Forward pass for the feed-forward network
        """
        x = linear(x, self.w1)  # [batch_size, seq_len, 4*dim]
        x = gelu(x)
        x = linear(x, self.w2)  # [batch_size, seq_len, dim]
        return x


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer model
    """
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)

        # Layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(config),
                'ln1': LayerNormModule(config.dim),
                'ffn': FeedForward(config),
                'ln2': LayerNormModule(config.dim)
            }) for _ in range(config.num_layers)
        ])

        # Output projection
        self.output_projection = nn.Parameter(self.token_embedding.weight.clone())

    def forward(self, input_ids: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        """
        Forward pass for the decoder-only Transformer model
        """
        # input_ids: [batch_size, seq_len]
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, dim]

        for layer in self.layers:
            # Self-attention with residual
            residual = x
            x = layer['ln1'](x)  # [batch_size, seq_len, dim]
            x = layer['attention'](x, use_cache=use_cache)  # [batch_size, seq_len, dim]
            x = x + residual  # [batch_size, seq_len, dim]

            # FFN with residual
            residual = x
            x = layer['ln2'](x)  # [batch_size, seq_len, dim]
            x = layer['ffn'](x)  # [batch_size, seq_len, dim]
            x = x + residual  # [batch_size, seq_len, dim]

        # Project to vocabulary
        logits = linear(x, self.output_projection)  # [batch_size, seq_len, vocab_size]
        return logits


class SimpleTextDataset(Dataset):
    """
    Simple text dataset for the Transformer model
    """
    def __init__(self, vocab_size, seq_len, size=1000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))
        # Shift sequences by 1 to create targets (next token prediction)
        self.targets = torch.roll(self.data, shifts=-1, dims=1)
        self.targets[:, -1] = 0  # pad last token
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


def train_epoch(model, dataloader, optimizer, device="cpu"):
    """
    Train the Transformer model for one epoch
    """
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    for batch_idx, (data, targets) in enumerate(dataloader):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(data)
        
        # Reshape logits and targets for loss calculation
        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def train_transformer():
    """
    Train the Transformer model
    """
    print("\nTraining Custom Transformer:")
    
    # Configuration
    config = TransformerConfig(
        vocab_size=1000,
        max_seq_len=64,
        dim=256,
        num_layers=2,
        num_heads=4
    )
    
    # Create model
    model = DecoderOnlyTransformer(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Create dataset and dataloader
    dataset = SimpleTextDataset(
        vocab_size=config.vocab_size,
        seq_len=32,
        size=500
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train for one epoch
    print(f"\nTraining on {device}")
    epoch_loss = train_epoch(model, dataloader, optimizer, device)
    print(f"\nEpoch completed. Average loss: {epoch_loss:.4f}")
    
    return model, epoch_loss


if __name__ == "__main__":
    train_transformer()