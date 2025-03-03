import torch
import torch.nn as nn
import numpy as np

class VisionTransformer(nn.Module):
    def __init__(self, num_heads, feedforward_dim, num_layers, unique_patches, 
                 max_patches, dropout=0.0, hidden_dim=None, num_partial_masks=32):
        super().__init__()
        # For 4x4 patches, embed_dim is 16
        embed_dim = 16
        
        # Store unique patches information
        self.unique_patches = unique_patches  # List of unique patch tensors
        
        # Calculate number of tokens: unique patches + 1 mask token + partial masks
        self.num_tokens = len(unique_patches) + 1 + num_partial_masks
        
        # Create the embedding matrix
        self.embedding_matrix = self._create_embedding_matrix(embed_dim, num_partial_masks)
        
        # Load hidden dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else 64
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(self.hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Linear layer to project input embeddings
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim)
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(max_patches, 1, self.hidden_dim))
        
        # Output layer
        self.fc_out = nn.Linear(self.hidden_dim, self.num_tokens - 1)  # -1 because we don't predict the mask token
    
    def _create_embedding_matrix(self, embed_dim, num_partial_masks):
        """Create embedding matrix for unique patches + mask token + partial masks"""
        # Initialize embedding matrix
        embedding_matrix = torch.zeros((self.num_tokens, embed_dim))
        
        # First, add embeddings for all unique patches
        for i, patch in enumerate(self.unique_patches):
            embedding_matrix[i] = patch
        
        # Add embedding for the mask token (at index len(unique_patches))
        mask_token_idx = len(self.unique_patches)
        embedding_matrix[mask_token_idx] = torch.tensor([0.5] * embed_dim)
        
        # Add embeddings for partially masked tokens
        # We'll focus on the case where one position is known (as you requested)
        partial_mask_start_idx = mask_token_idx + 1
        partial_mask_count = 0
        
        # Add partial masks systematically (up to the limit)
        for position in range(min(embed_dim, 16)):  # Only create partial masks for first 16 positions if needed
            for value in [0, 1]:  # Binary values
                if partial_mask_count >= num_partial_masks:
                    break
                    
                idx = partial_mask_start_idx + partial_mask_count
                # Create a tensor of 0.5s with one position set to the known value
                patch_values = torch.tensor([0.5] * embed_dim)
                patch_values[position] = float(value)
                embedding_matrix[idx] = patch_values
                partial_mask_count += 1
        
        return nn.Embedding.from_pretrained(embedding_matrix, freeze=True)

    def forward(self, patches):
        # Retrieve embeddings
        embeddings = self.embedding_matrix(patches)  # (batch_size, seq_len, embed_dim)

        # Prepare input for transformer layers
        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # Extract seq_len and batch size
        seq_len, batch_size, _ = x.size()
        
        # Project input to hidden_dim
        z = self.input_proj(x)  # Shape: (seq_len, batch_size, hidden_dim)

        # Add positional embedding
        pos_emb = self.positional_embedding[:seq_len, :, :].expand(-1, batch_size, -1)
        z = z + pos_emb

        # Pass through transformer layers
        for layer in self.encoder_layers:
            z = layer(z)

        # Output logits
        z = z.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_dim)
        logits = self.fc_out(z)  # (batch_size, seq_len, num_tokens-1)
        return logits

    def get_probabilities(self, logits):
        """Compute probabilities using softmax."""
        return torch.softmax(logits, dim=-1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, "Hidden dimension must be divisible by the number of heads."

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, self.hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # Apply LayerNorm
        z_norm = self.norm1(z)

        # Self-attention
        attn_output, _ = self.attention(z_norm, z_norm, z_norm)

        # Residual connection
        z = z + self.dropout(attn_output)

        # Feedforward layer
        z_norm = self.norm2(z)
        feedforward_output = self.feedforward(z_norm)

        # Final residual connection
        z = z + self.dropout(feedforward_output)

        return z
