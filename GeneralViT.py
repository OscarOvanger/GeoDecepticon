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
    # Assuming training_data shape is (N, H, W)
    H, W = training_data.shape[1], training_data.shape[2]
    num_patches = (H // patch_size) * (W // patch_size)
    
    # Extract non-overlapping patches from all images at once.
    # The unfold operations extract patches along height and width.
    patches = training_data.unfold(1, patch_size, patch_size) \
                           .unfold(2, patch_size, patch_size)
    # patches shape: (N, num_patches_h, num_patches_w, patch_size, patch_size)
    # Reshape to (total_patches, patch_dim)
    patches = patches.contiguous().view(-1, patch_dim)
    
    # Find unique patches
    unique_patches = torch.unique(patches, dim=0)
    
    # Build vocabulary list from unique patches
    vocab_list = [patch for patch in unique_patches]
    
    # Add full mask token if needed
    if full_mask:
        vocab_list.append(torch.full((patch_dim,), 0.5, dtype=torch.float))
    
    # Add one-mask tokens for each patch dimension if needed
    if one_mask:
        for i in range(patch_dim):
            mask_patch = torch.full((patch_dim,), 0.5, dtype=torch.float)
            mask_patch[i] = 0.0
            vocab_list.append(mask_patch)
            
            mask_patch = torch.full((patch_dim,), 0.5, dtype=torch.float)
            mask_patch[i] = 1.0
            vocab_list.append(mask_patch)
    
    # Initialize vocabulary and related model components
    self.vocab = torch.stack(vocab_list)
    self.vocab_size = self.vocab.size(0)
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
    
    
  
