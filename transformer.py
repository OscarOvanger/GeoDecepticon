import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, num_tokens, max_patches, dropout=0.0):
        super().__init__()
        # Create the embedding matrix for all 2x2 binary combinations + 1 mask token
        embedding_matrix = torch.zeros((num_tokens, embed_dim))  # Shape: (num_tokens, embed_dim)

        # Generate all possible 2x2 binary patches
        patches = torch.tensor([
            [a, b, c, d]
            for a in range(2)
            for b in range(2)
            for c in range(2)
            for d in range(2)
        ])  # Shape: (16, 4) for 16 combinations of 2x2 patches

        # Assign each patch's values as its embedding
        for i, patch in enumerate(patches):
            embedding_matrix[i, :] = patch  # Set the embedding to the patch values

        # Set the last row to all 2s for the masked patch
        embedding_matrix[-1, :] = 2  # Mask token embedding

        # Create embedding layer
        self.embedding_matrix = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True
        )

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(embed_dim, num_tokens)

    def forward(self, patches, mask):
        # Retrieve embeddings
        embeddings = self.embedding_matrix(patches)

        # Replace masked patches with mask token embedding
        # This step might actually not do anything, but makes sure that no funny business is going on
        mask_token_embedding = self.embedding_matrix.weight[-1]
        embeddings[mask.bool()] = mask_token_embedding

        # Prepare input for transformer layers
        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # Pass through transformer layers with mask
        for layer in self.encoder_layers:
            x = layer(x, mask)

        # Output logits
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_dim)
        logits = self.fc_out(x)  # (batch_size, seq_len, num_tokens)
        return logits

    def get_probabilities(self, logits):
        """Compute probabilities using softmax."""
        return torch.softmax(logits, dim=-1)

import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, max_relative_positions=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_relative_positions = max_relative_positions

        # Ensure embed_dim is divisible by num_heads
        assert (
            self.embed_dim % self.num_heads == 0
        ), "Embedding dimension must be divisible by the number of heads."

        # Self-attention weights
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)  # For queries, keys, and values
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Relative positional embeddings
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(2 * max_relative_positions - 1, self.head_dim)
        )

    def compute_relative_positions(self, seq_len):
        """
        Compute the relative position matrix for a sequence of length `seq_len`.
        """
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]  # Shape: (seq_len, seq_len)
        relative_positions += self.max_relative_positions - 1  # Shift to positive range
        relative_positions = relative_positions.clamp(0, 2 * self.max_relative_positions - 2)  # Bound indices
        return relative_positions

    def add_relative_position_scores(self, attention_scores, seq_len, batch_size):
        """
        Add relative positional embeddings to the attention scores.
    
        Args:
            attention_scores: Tensor of shape (batch_size, num_heads, seq_len, seq_len)
            seq_len: Sequence length
            batch_size: Batch size
    
        Returns:
            Updated attention scores with relative positional embeddings
        """
        # Compute relative positions
        relative_positions = self.compute_relative_positions(seq_len)  # Shape: (seq_len, seq_len)
    
        # Retrieve corresponding relative positional embeddings
        relative_position_scores = self.relative_position_embeddings[relative_positions]  # Shape: (seq_len, seq_len, head_dim)
    
        # Sum over the embedding dimension (head_dim)
        relative_position_scores = relative_position_scores.sum(dim=-1)  # Shape: (seq_len, seq_len)
    
        # Expand for batch and num_heads
        relative_position_scores = relative_position_scores.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, seq_len)
        relative_position_scores = relative_position_scores.expand(batch_size, self.num_heads, -1, -1)  # Shape: (batch_size, num_heads, seq_len, seq_len)
    
        # Add relative positional scores to attention scores
        attention_scores += relative_position_scores
        return attention_scores

    def forward(self, x, mask=None):
        seq_len, batch_size, embed_dim = x.size()
    
        # Create key padding mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # Invert mask
    
        # Apply LayerNorm
        x_norm = self.norm1(x)
    
        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm).view(seq_len, batch_size, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # Shape: (seq_len, batch_size, num_heads, head_dim)
    
        # Transpose for multi-head attention compatibility
        q = q.permute(1, 2, 0, 3)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.permute(1, 2, 0, 3)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.permute(1, 2, 0, 3)  # (batch_size, num_heads, seq_len, head_dim)
    
        # Compute attention scores
        attention_scores = torch.einsum("bhqd,bhkd->bhqk", q, k) / (self.head_dim ** 0.5)  # (batch_size, num_heads, seq_len, seq_len)
    
        # Add relative positional embeddings to scores
        attention_scores = self.add_relative_position_scores(attention_scores, seq_len, batch_size)
    
        # Apply mask (optional)
        if mask is not None:
            expanded_mask = mask.unsqueeze(1).unsqueeze(2)  # Shape: (batch_size, 1, 1, seq_len)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, float("-inf"))
    
        # Compute attention probabilities
        attention_probs = F.softmax(attention_scores, dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
    
        # Compute attention output
        attention_output = torch.einsum("bhqk,bhkd->bhqd", attention_probs, v)  # (batch_size, num_heads, seq_len, head_dim)
    
        # Concatenate heads
        attention_output = attention_output.permute(2, 0, 1, 3).reshape(seq_len, batch_size, embed_dim)  # (seq_len, batch_size, embed_dim)
    
        # Apply output projection
        attention_output = self.out_proj(attention_output)
    
        # Residual connection
        x = x + self.dropout(attention_output)
    
        # Feedforward layer
        x_norm = self.norm2(x)
        feedforward_output = self.feedforward(x_norm)
        x = x + self.dropout(feedforward_output)
    
        return x
