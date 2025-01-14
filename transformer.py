import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, num_tokens, max_patches, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.embedding_matrix = nn.Embedding(num_tokens + 1, embed_dim)  # +1 for masked token
        self.embedding_matrix.weight.requires_grad = False  # Fixed weights

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_dim, num_tokens)

    def forward(self, patches, mask):
        embeddings = self.embedding_matrix(patches)
        embeddings[mask == 1] = self.embedding_matrix.weight[-1]  # Mask token embedding

        x = embeddings.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        for layer in self.layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # (batch_size, seq_len, embed_dim)

        logits = self.fc_out(x)
        return logits

    def get_probabilities(self, logits):
        """
        Computes probabilities using softmax for all patches.
        Args:
            logits (Tensor): Logits from the model, shape (batch_size, num_patches, num_tokens).

        Returns:
            probabilities (Tensor): Probabilities for each token, shape (batch_size, num_patches, num_tokens).
        """
        return torch.softmax(logits, dim=-1)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
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