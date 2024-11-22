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

        # MultiheadAttention layer
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=False)

        # Feedforward layer
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
            torch.randn((2 * max_relative_positions - 1), self.head_dim)  # One for each relative position
        )

    def compute_relative_positions(self, seq_len):
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]  # (seq_len, seq_len)
        relative_positions = relative_positions + self.max_relative_positions - 1
        relative_positions = relative_positions.clamp(0, 2 * self.max_relative_positions - 2)  # Stay within bounds
        return relative_positions

    def add_relative_position_scores(self, attention_scores, seq_len):
        """
        Add relative positional embeddings to attention scores.

        Args:
            attention_scores (Tensor): (batch_size * num_heads, seq_len, seq_len).
            seq_len (int): Sequence length.

        Returns:
            Tensor: Updated attention scores.
        """
        # Compute relative positions
        relative_positions = self.compute_relative_positions(seq_len)  # (seq_len, seq_len)

        # Retrieve relative positional embeddings
        relative_position_scores = self.relative_position_embeddings[relative_positions]  # (seq_len, seq_len, head_dim)

        # Sum along the head dimension to match attention scores shape
        relative_position_scores = relative_position_scores.sum(dim=-1)  # (seq_len, seq_len)

        # Add relative positional scores
        attention_scores += relative_position_scores
        return attention_scores

    def forward(self, x, mask=None):
        """
        Forward pass for a single Transformer encoder layer.
        
        Args:
            x (Tensor): Input embeddings, shape (seq_len, batch_size, embed_dim).
            mask (Tensor): Optional key padding mask, shape (batch_size, seq_len).

        Returns:
            Tensor: Output embeddings, shape (seq_len, batch_size, embed_dim).
        """
        seq_len, batch_size, embed_dim = x.size()

        # Compute key padding mask for attention
        key_padding_mask = None
        if mask is not None:
            key_padding_mask = ~mask.bool()  # Invert mask for valid positions

        # Multi-head attention
        attn_output, attn_weights = self.attention(x, x, x, key_padding_mask=key_padding_mask, need_weights=True)

        # Add relative positional embeddings to attention scores
        attn_output = self.add_relative_position_scores(attn_output, seq_len)

        # Residual connection and normalization
        x = self.norm1(x + attn_output)

        # Feedforward layer
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)

        return x
