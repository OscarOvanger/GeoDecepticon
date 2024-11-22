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
        # We create the embedding layer with freeze = True, because we fix it in this case.
        self.embedding_matrix = nn.Embedding.from_pretrained(
            embedding_matrix, freeze=True
        )
        
        # Learnable positional embeddings
        #self.positional_embeddings = nn.Parameter(
        #    torch.randn(1, max_patches, embed_dim)  # Shape: (1, num_patches, embed_dim)
        #)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        # Output layer
        self.fc_out = nn.Linear(embed_dim, num_tokens)

    def forward(self, patches, mask):
        # Retrieve embeddings
        # This is the input to the transformer
        embeddings = self.embedding_matrix(patches)  # (batch_size, num_patches, embed_dim)

        # Replace masked patches with mask token embedding
        mask_token_embedding = self.embedding_matrix.weight[-1]
        embeddings[mask.bool()] = mask_token_embedding

        # Add positional embeddings
        # We add positional embeddings to the Inputs
        #embeddings = embeddings + self.positional_embeddings  # Add positional info

        # Transformer encoder layers
        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        for layer in self.encoder_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # Back to (batch_size, seq_len, embed_dim)

        # Output logits
        # This is x[i,j,:]*W_vocab + b_bias, it is logits that can later be taken softmax of in the crossentropy loss.
        logits = self.fc_out(x)  # (batch_size, seq_len, num_tokens)
        return logits

    def get_probabilities(self, logits):
        """Compute probabilities using softmax."""
        return torch.softmax(logits, dim=-1)
"""
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)
        return x
"""
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1, max_relative_positions=32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_relative_positions = max_relative_positions

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
            torch.randn((2 * max_relative_positions - 1), embed_dim // num_heads)
        )

    def compute_relative_positions(self, seq_len):
        # Create a relative position matrix
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]  # Shape: (seq_len, seq_len)
        relative_positions = relative_positions + self.max_relative_positions - 1
        relative_positions = relative_positions.clamp(0, 2 * self.max_relative_positions - 2)  # Keep within bounds
        return relative_positions

    def add_relative_position_scores(self, attention_scores, seq_len):
        # Compute relative position embeddings
        relative_positions = self.compute_relative_positions(seq_len)
        relative_position_scores = self.relative_position_embeddings[relative_positions]  # (seq_len, seq_len, head_dim)

        # Add relative position scores to the attention scores
        relative_position_scores = relative_position_scores.permute(2, 0, 1)  # (head_dim, seq_len, seq_len)
        attention_scores = attention_scores + relative_position_scores  # Add to attention scores
        return attention_scores

    def forward(self, x):
        seq_len, batch_size, embed_dim = x.size()

        # Multi-head attention with relative position embeddings
        attn_output, _ = self.attention(x, x, x)  # Standard attention
        attn_output = self.add_relative_position_scores(attn_output, seq_len)  # Add relative position scores

        # Residual connection and normalization
        x = self.norm1(x + attn_output)

        # Feedforward
        feedforward_output = self.feedforward(x)
        x = self.norm2(x + feedforward_output)

        return x
    

# Test the Vision Transformer
def test_transformer():
    print("Testing Vision Transformer...")

    # Define dummy input
    batch_size = 2
    num_patches = 32 * 32  # For 64x64 image with 2x2 patches
    embed_dim = 4
    num_tokens = 16
    masked_token_index = num_tokens  # Mask token index

    # Create dummy input data
    input_patches = torch.randint(0, num_tokens, (batch_size, num_patches))
    mask = torch.zeros_like(input_patches, dtype=torch.bool)
    mask[:, :100] = 1  # Mask the first 100 patches

    # Initialize the model
    model = VisionTransformer(embed_dim, num_heads=2, feedforward_dim=8, num_layers=2, num_tokens=num_tokens, max_patches=num_patches)

    # Forward pass
    logits = model(input_patches, mask)
    print("logits:", logits)
    assert logits.shape == (batch_size, num_patches, num_tokens), "Logits shape is incorrect"
    print("Logits shape test passed.")

    # Check probabilities
    probabilities = model.get_probabilities(logits)
    print("probabilities:", probabilities)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones_like(probabilities.sum(dim=-1)), atol=1e-5), "Probabilities do not sum to 1"
    print("Softmax probabilities test passed.")

#test_transformer()
