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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, max_relative_positions=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_relative_positions = max_relative_positions
        self.head_dim = embed_dim // num_heads

        # Attention and feedforward layers
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Relative positional embeddings
        self.relative_position_embeddings = nn.Parameter(
            torch.randn((2 * max_relative_positions - 1), self.head_dim)
        )

    def compute_relative_positions(self, seq_len):
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]  # (seq_len, seq_len)
        relative_positions = relative_positions + self.max_relative_positions - 1
        relative_positions = relative_positions.clamp(0, 2 * self.max_relative_positions - 2)
        return relative_positions

    def add_relative_position_scores(self, attention_scores, seq_len):
    """
    Add relative positional embeddings to the attention scores.

    Args:
        attention_scores (Tensor): Attention scores, shape (batch_size * num_heads, seq_len, seq_len).
        seq_len (int): Length of the sequence.

    Returns:
        Tensor: Updated attention scores with relative positional embeddings.
    """
        # Compute relative positions
        relative_positions = self.compute_relative_positions(seq_len)  # Shape: (seq_len, seq_len)
    
        # Retrieve corresponding relative positional embeddings
        relative_position_scores = self.relative_position_embeddings[relative_positions]  # Shape: (seq_len, seq_len, head_dim)
    
        # Expand and reshape to match attention scores
        relative_position_scores = relative_position_scores.permute(2, 0, 1)  # Shape: (head_dim, seq_len, seq_len)
        relative_position_scores = relative_position_scores.unsqueeze(0).expand(
            self.num_heads, -1, -1, -1
        )  # Shape: (num_heads, head_dim, seq_len, seq_len)
        relative_position_scores = relative_position_scores.sum(dim=1)  # Sum over head_dim: (num_heads, seq_len, seq_len)
    
        # Expand to match attention_scores (batch_size * num_heads, seq_len, seq_len)
        relative_position_scores = relative_position_scores.repeat_interleave(
            attention_scores.size(0) // self.num_heads, dim=0
        )  # Shape: (batch_size * num_heads, seq_len, seq_len)
    
        # Add relative positional scores
        attention_scores = attention_scores + relative_position_scores
        return attention_scores

    def forward(self, x, mask=None):
        seq_len, batch_size, embed_dim = x.size()

        # Create key padding mask
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # Invert mask

        # Multi-head attention
        q, k, v = x, x, x
        attention_scores = self.attention(q, k, v, key_padding_mask=key_padding_mask)[0]

        # Add relative positional embeddings to scores
        attention_scores = self.add_relative_position_scores(attention_scores, seq_len)

        # Apply residual connection and normalization
        x = self.norm1(x + attention_scores)

        # Feedforward layer
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)

        return x
