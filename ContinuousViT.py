import torch
import torch.nn as nn
import torch.nn.functional as F

class ContinuousVisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, max_patches, dropout=0.1):
        super().__init__()
        # Constants
        self.patch_dim = 16  # 4x4 binary patches
        
        # Core components
        self.patch_projection = nn.Linear(self.patch_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(1, embed_dim))
        self.partial_mask_indicator = nn.Parameter(torch.randn(1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.patch_decoder = nn.Linear(embed_dim, self.patch_dim)
    
    def forward(self, patches, mask_indices=None, partial_mask_indices=None, partial_mask_values=None):
        """
        Forward pass with simplified masking logic.
        
        Args:
            patches: Tensor [batch_size, num_patches, patch_dim]
            mask_indices: Boolean tensor for fully masked patches [batch_size, num_patches]
            partial_mask_indices: Boolean tensor for partially masked patches [batch_size, num_patches]
            partial_mask_values: Tensor with partial values [batch_size, num_partial, patch_dim]
        
        Returns:
            dict: Model outputs including logits and probabilities
        """
        batch_size, num_patches, _ = patches.shape
        
        # Project patches to embeddings
        patch_embeddings = self.patch_projection(patches)
        
        # Apply full masking - replace with mask token
        if mask_indices is not None:
            # Expand mask token to match batch dimension
            mask_token_expanded = self.mask_token.expand(batch_size, -1)  # [batch_size, embed_dim]
            
            # Use boolean indexing to apply mask token
            for b in range(batch_size):
                patch_embeddings[b][mask_indices[b]] = mask_token_expanded[b]
        
        # Apply partial masking
        if partial_mask_indices is not None and partial_mask_values is not None:
            for b in range(batch_size):
                partial_idx = torch.where(partial_mask_indices[b])[0]
                for i, idx in enumerate(partial_idx):
                    if i < partial_mask_values.size(1):
                        # Project partial values
                        partial_proj = self.patch_projection(partial_mask_values[b, i])
                        # Add partial mask indicator
                        patch_embeddings[b, idx] = partial_proj + self.partial_mask_indicator
        
        # Add position embeddings
        embeddings = patch_embeddings + self.pos_embedding[:, :num_patches]
        
        # Process through transformer
        x = embeddings.permute(1, 0, 2)  # [num_patches, batch, embed_dim]
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # [batch, num_patches, embed_dim]
        
        # Decode to patch values
        logits = self.patch_decoder(x)
        probs = torch.sigmoid(logits)
        binary_pred = (probs > 0.5).float()
        
        return {
            'logits': logits,
            'probabilities': probs,
            'binary_prediction': binary_pred,
            'embeddings': x
        }
    
    def get_loss(self, outputs, original_patches, mask_indices=None, partial_mask_indices=None, partial_mask_values=None):
        """
        Calculate binary cross-entropy loss on masked positions only.
        """
        batch_size, num_patches, patch_dim = original_patches.shape
        device = original_patches.device
        logits = outputs['logits']
        
        # Create pixel-level mask for loss calculation
        pixel_mask = torch.zeros((batch_size, num_patches, patch_dim), dtype=torch.bool, device=device)
        
        # For fully masked patches, mask all pixels
        if mask_indices is not None:
            for b in range(batch_size):
                pixel_mask[b, mask_indices[b]] = True
        
        # For partially masked patches, only mask the unknown positions (0.5)
        if partial_mask_indices is not None and partial_mask_values is not None:
            for b in range(batch_size):
                partial_idx = torch.where(partial_mask_indices[b])[0]
                for i, idx in enumerate(partial_idx):
                    if i < partial_mask_values.size(1):
                        # Mask only unknown positions (marked as 0.5)
                        unknown_pos = (partial_mask_values[b, i] == 0.5)
                        pixel_mask[b, idx, unknown_pos] = True
        
        # Extract masked positions
        if pixel_mask.sum() > 0:
            masked_logits = logits[pixel_mask]
            masked_targets = original_patches[pixel_mask]
            
            # Calculate loss
            loss = F.binary_cross_entropy_with_logits(
                masked_logits, 
                masked_targets,
                reduction='mean'
            )
            return loss
        else:
            # No pixels to predict
            return torch.tensor(0.0, device=device, requires_grad=True)
    
    def reconstruct_image(self, patch_values):
        """
        Reconstruct 64x64 image from patch values.
        """
        patch_size = 4
        image_size = 64
        patches_per_dim = image_size // patch_size
        
        # Initialize output image
        image = torch.zeros((image_size, image_size), device=patch_values.device)
        
        # Place each patch in the image
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                idx = i * patches_per_dim + j
                if idx < patch_values.size(0):
                    patch = patch_values[idx].reshape(patch_size, patch_size)
                    image[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patch
        
        return image


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
            nn.GELU(),
            nn.Linear(feedforward_dim, self.hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # Apply LayerNorm before attention (pre-norm formulation)
        z_norm = self.norm1(z)
        
        # Self-attention
        attn_output, _ = self.attention(z_norm, z_norm, z_norm)
        
        # Residual connection
        z = z + self.dropout(attn_output)
        
        # Apply LayerNorm before feedforward
        z_norm = self.norm2(z)
        
        # Feedforward layer
        feedforward_output = self.feedforward(z_norm)
        
        # Final residual connection
        z = z + self.dropout(feedforward_output)
        
        return z
