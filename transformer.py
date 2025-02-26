import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, num_tokens, max_patches, dropout=0.0, hidden_dim=None):
        super().__init__()
        # Create the embedding matrix for all 2x2 binary combinations + 1 mask token
        embedding_matrix = torch.zeros((num_tokens, embed_dim))  # Shape: (num_tokens, embed_dim)
        
        #Load hidden dim
        self.hidden_dim = hidden_dim
        
        # Generate all possible 2x2 binary patches
        patches = torch.tensor([
            [a, b, c, d]
            for a in range(3)
            for b in range(3)
            for c in range(3)
            for d in range(3)
        ])  # Shape: (81, 4) for 81 combinations of 2x2 patches

        # Assign each patch's values as its embedding
        for i, patch in enumerate(patches):
            patch = torch.where(patch == 2.0,0.5,patch)
            embedding_matrix[i, :] = patch  # Set the embedding to the patch values

        # Create embedding layer
        self.embedding_matrix = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True
        )

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        # Linear layer to project input embeddings
        self.input_proj = nn.Linear(embed_dim, self.hidden_dim)
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(max_patches, 1, self.hidden_dim))  # Shape: (seq_len, 1, hidden_dim)
        
        # Output layer
        self.fc_out = nn.Linear(self.hidden_dim, 2**embed_dim)

    def forward(self, patches):
        # Retrieve embeddings
        embeddings = self.embedding_matrix(patches)  # (batch_size, seq_len, embed_dim)

        # Prepare input for transformer layers
        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        #Extract seq_len and batch size
        seq_len,batch_size,_ = x.size()
        # Project input to hidden_dim
        z = self.input_proj(x)  # Shape: (seq_len, batch_size, hidden_dim)

        # Add positional embedding
        pos_emb = self.positional_embedding[:seq_len, :, :].expand(-1, batch_size, -1)  # Shape: (seq_len, batch_size, hidden_dim)
        z = z + pos_emb

        # Pass through transformer layers
        for layer in self.encoder_layers:
            z = layer(z)

        # Output logits
        z = z.permute(1, 0, 2)  # Back to (batch_size, seq_len, hidden_dim)
        logits = self.fc_out(z)  # (batch_size, seq_len, num_tokens)
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
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, embed_dim)
        Returns:
            Tensor of shape (seq_len, batch_size, embed_dim)
        """
        seq_len, batch_size, hidden_dim = z.size()

        # Apply LayerNorm
        z_norm = self.norm1(z)

        # Self-attention
        attn_output, _ = self.attention(z_norm, z_norm, z_norm)  # Shape: (seq_len, batch_size, hidden_dim)

        # Residual connection
        z = z + self.dropout(attn_output)

        # Feedforward layer
        z_norm = self.norm2(z)
        feedforward_output = self.feedforward(z_norm)

        # Final residual connection
        z = z + self.dropout(feedforward_output)

        return z
