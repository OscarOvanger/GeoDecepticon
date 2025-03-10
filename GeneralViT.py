import torch
import torch.nn as nn

class VisionTransformer(nn.Module):
  def __init__(self, num_heads, num_layers, ffn_dim, hidden_dim, dropout=0.0):
    super().__init__()
    self.hidden_dim = hidden_dim
    self.transformer_layers = nn.ModuleList([
        TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout)
        for _ in range(num_layers)
    ])
    self.dropout_layer = nn.Dropout(dropout)
        
  def build_vocabulary(self, training_data, patch_size, full_mask=True, one_mask=True):
    self.patch_size = patch_size
    patch_dim = patch_size * patch_size
    img_size = training_data.shape[1]*training_data.shape[2]  # Assuming shape (batch, height, width)
    num_patches = (img_size // patch_size) ** 2
    # Collect unique patches
    unique_patches = set()
    for img in training_data:
        if img.dim() == 2:
            for i in range(0, img.size(0) - patch_size + 1, patch_size):
                for j in range(0, img.size(1) - patch_size + 1, patch_size):
                    patch = img[i:i+patch_size, j:j+patch_size].reshape(-1)
                    unique_patches.add(tuple(patch.tolist()))
        else:
            for b in range(img.size(0)):
                for i in range(0, img.size(1) - patch_size + 1, patch_size):
                    for j in range(0, img.size(2) - patch_size + 1, patch_size):
                        patch = img[b, i:i+patch_size, j:j+patch_size].reshape(-1)
                        unique_patches.add(tuple(patch.tolist()))
        
    # Create vocabulary
    vocab = [torch.tensor(patch, dtype=torch.float) for patch in unique_patches]
        
    # Add masked patches
    if full_mask:
        vocab.append(torch.tensor([0.5] * patch_dim, dtype=torch.float))
        
    if one_mask:
        for i in range(patch_dim):
            masked_0 = torch.tensor([0.5] * patch_dim, dtype=torch.float)
            masked_0[i] = 0.0
            vocab.append(masked_0)
                
            masked_1 = torch.tensor([0.5] * patch_dim, dtype=torch.float)
            masked_1[i] = 1.0
            vocab.append(masked_1)
        
    # Initialize model components
    self.vocab = torch.stack(vocab)
    self.vocab_size = len(self.vocab)
    self.embedding_projection = nn.Linear(patch_dim, self.hidden_dim)
    self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)
    self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, self.hidden_dim))
        
    return self.vocab
        
  def forward(self, patches):
    # Prepare input for Transformer Layer
    batch_size, num_patches, _ = patches.shape
        
    # Project input to hidden_dim
    embeddings = self.embedding_projection(patches)
        
    # Add Positional Embedding
    embeddings = embeddings + self.pos_embedding[:, :num_patches, :]
        
    # Pass Through transformer layers
    z = self.dropout_layer(embeddings)
    for layer in self.transformer_layers:
        z = layer(z)
        
    # Output logits
    logits = self.output_projection(z)
        
    return logits

class TransformerEncoderLayer(nn.Module):
  def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
    self.linear1 = nn.Linear(hidden_dim, ffn_dim)
    self.linear2 = nn.Linear(ffn_dim, hidden_dim)
    self.norm1 = nn.LayerNorm(hidden_dim)
    self.norm2 = nn.LayerNorm(hidden_dim)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.activation = nn.GELU()
        
  def forward(self, z):
    # Apply LayerNorm
    z_norm = self.norm1(z)
        
    # Self Attention
    z_t = z_norm.transpose(0, 1)
    attn_output, _ = self.self_attn(z_t, z_t, z_t)
    attn_output = attn_output.transpose(0, 1)
        
    # Residual connection
    z = z + self.dropout1(attn_output)
        
    # FeedForward Layer
    z_norm = self.norm2(z)
    ff_output = self.linear2(self.dropout2(self.activation(self.linear1(z_norm))))
        
    # Final Residual connection
    z = z + self.dropout2(ff_output)
        
    return z

def create_model(train_dataset, num_heads, num_layers, ffn_dim, hidden_dim, patch_size):
    """
    Create and initialize a Vision Transformer model
    """
    model = VisionTransformer(num_heads, num_layers, ffn_dim, hidden_dim, dropout=0.0)
    # Build vocabulary from training data
    model.build_vocabulary(train_dataset, patch_size)
    
    return model
    
    
  
