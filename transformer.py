import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, num_tokens, max_patches, dropout=0.0, hidden_dim=None):
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
        embedding_matrix[-1, :] = 0.5  # Mask token embedding

        # Create embedding layer
        self.embedding_matrix = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True
        )

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout, hidden_dim=hidden_dim)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(embed_dim, num_tokens)

    def forward(self, patches):
        # Retrieve embeddings
        embeddings = self.embedding_matrix(patches)  # (batch_size, seq_len, embed_dim)

        # Prepare input for transformer layers
        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Output logits
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_dim)
        logits = self.fc_out(x)  # (batch_size, seq_len, num_tokens)
        return logits

    def get_probabilities(self, logits):
        """Compute probabilities using softmax."""
        return torch.softmax(logits, dim=-1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, hidden_dim=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim if hidden_dim else embed_dim  # Default to embed_dim if hidden_dim is not specified
        self.num_heads = num_heads

        # Linear layer to project input embeddings
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim)

        # Positional embeddings
        self.positional_embeddings = nn.Parameter(torch.randn(1, self.hidden_dim))

        # Multi-head attention
        self.multihead_attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len, batch_size, _ = x.size()

        # Project input to hidden_dim
        z = self.input_proj(x)  # Shape: (seq_len, batch_size, hidden_dim)

        # Add learnable positional embeddings
        z = z + self.positional_embeddings  # Broadcasting to (seq_len, batch_size, hidden_dim)

        # Apply LayerNorm before attention
        z_norm = self.norm1(z)

        # Multi-head attention
        attention_output, _ = self.multihead_attention(z_norm, z_norm, z_norm)  # Shape: (seq_len, batch_size, hidden_dim)

        # Residual connection
        z = z + self.dropout(attention_output)

        # Project back to embed_dim
        z = self.input_proj(z)

        # Feedforward layer with residual connection
        z_norm = self.norm2(z)
        feedforward_output = self.feedforward(z_norm)
        x = z + self.dropout(feedforward_output)

        return x
