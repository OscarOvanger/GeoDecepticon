import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ContinuousVisionTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, feedforward_dim, num_layers, max_patches, dropout=0.1):
        super().__init__()
        # Patch dimension for 4x4 binary patches
        patch_dim = 16
        
        # Patch embedding projection
        self.patch_projection = nn.Linear(patch_dim, embed_dim)
        
        # Learnable mask token embedding
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Learnable partial mask indicator embedding
        self.partial_mask_indicator = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(1, max_patches, embed_dim))
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Patch reconstruction head
        self.patch_decoder = nn.Linear(embed_dim, patch_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights for better convergence"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, patches, mask_indices=None, partial_mask_indices=None, partial_mask_values=None):
        """
        Forward pass with support for full and partial masking
        
        Args:
            patches: Tensor of shape [batch_size, num_patches, 16] with patch pixel values
            mask_indices: Boolean tensor indicating fully masked patches
            partial_mask_indices: Boolean tensor indicating partially masked patches
            partial_mask_values: Tensor with values for partial masks
        
        Returns:
            outputs: Dictionary containing model outputs
        """
        batch_size, num_patches, patch_dim = patches.shape
        device = patches.device
        
        # IMPORTANT: Create a copy of patches to avoid in-place modifications
        # that would break autograd
        patches_modified = patches.clone()
        
        # Create patch embeddings through linear projection
        patch_embeddings = self.patch_projection(patches_modified)
        
        # Apply masking if specified
        if mask_indices is not None:
            # Handle fully masked patches
            for b in range(batch_size):
                # Get indices of fully masked patches for this batch
                full_mask_idx = torch.where(mask_indices[b])[0]
                if len(full_mask_idx) > 0:
                    # Apply mask token
                    mask_token_expanded = self.mask_token.squeeze(0).expand(len(full_mask_idx), -1)
                    patch_embeddings[b, full_mask_idx] = mask_token_expanded
        
        # Apply partial masking if specified
        if partial_mask_indices is not None and partial_mask_values is not None:
            for b in range(batch_size):
                # Get indices of partially masked patches for this batch
                partial_mask_idx = torch.where(partial_mask_indices[b])[0]
                if len(partial_mask_idx) > 0:
                    # Process each partially masked patch
                    for i, idx in enumerate(partial_mask_idx):
                        if i < partial_mask_values.size(1):  # Safety check
                            # Get the partial mask pattern
                            partial_values = partial_mask_values[b, i]
                            
                            # Create a new patch with partial masking applied
                            # Instead of modifying patches in-place, we create a new tensor
                            mask_positions = (partial_values == 0.5)
                            partial_patch = torch.where(
                                mask_positions,
                                torch.zeros_like(partial_values),
                                partial_values
                            )
                            
                            # Project the modified patch
                            projected_patch = self.patch_projection(partial_patch)
                            
                            # Add the partial mask indicator
                            partial_indicator = self.partial_mask_indicator.squeeze(0).squeeze(0)
                            
                            # Update embedding
                            patch_embeddings[b, idx] = projected_patch + partial_indicator
        
        # Add position embeddings
        embeddings = patch_embeddings + self.pos_embedding[:, :num_patches]
        
        # Process through transformer layers
        x = embeddings.permute(1, 0, 2)  # [num_patches, batch, embed_dim]
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # [batch, num_patches, embed_dim]
        
        # Decode to patch values
        patch_logits = self.patch_decoder(x)
        
        # Apply sigmoid to get probabilities for each position
        patch_probs = torch.sigmoid(patch_logits)
        
        # For inference, round probabilities to binary values
        binary_pred = (patch_probs > 0.5).float()
        
        # Prepare outputs
        outputs = {
            'logits': patch_logits,
            'probabilities': patch_probs,
            'binary_prediction': binary_pred,
            'embeddings': x
        }
        
        return outputs
    
    def get_loss(self, outputs, original_patches, mask_indices=None, partial_mask_indices=None, partial_mask_values=None):
        """
        Calculate BCE loss for patch reconstruction, properly handling partial masking.
        
        Args:
            outputs: Dictionary from forward pass with 'logits'
            original_patches: Original patch values [batch, num_patches, patch_dim]
            mask_indices: Boolean tensor for fully masked patches [batch, num_patches]
            partial_mask_indices: Boolean tensor for partially masked patches [batch, num_patches]
            partial_mask_values: Tensor with values for partial masks [batch, num_partial, patch_dim]
                Values of 0.5 indicate masked positions, 0/1 indicate known positions
                
        Returns:
            loss: Binary cross entropy loss computed only on masked pixel positions
        """
        batch_size, num_patches, patch_dim = original_patches.shape
        device = original_patches.device
        logits = outputs['logits']
        
        # Initialize pixel-level mask
        pixel_mask = torch.zeros((batch_size, num_patches, patch_dim), dtype=torch.bool, device=device)
        
        # Handle fully masked patches - all pixels are masked
        if mask_indices is not None:
            for b in range(batch_size):
                full_mask_indices = torch.where(mask_indices[b])[0]
                if len(full_mask_indices) > 0:
                    # Mark all pixels in fully masked patches
                    pixel_mask[b, full_mask_indices] = True
        
        # Handle partially masked patches - only mask unknown positions
        if partial_mask_indices is not None and partial_mask_values is not None:
            for b in range(batch_size):
                partial_indices = torch.where(partial_mask_indices[b])[0]
                for i, patch_idx in enumerate(partial_indices):
                    if i < partial_mask_values.size(1):  # Safety check
                        # In partial masks, 0.5 indicates masked (unknown) positions
                        unknown_positions = (partial_mask_values[b, i] == 0.5)
                        # Only mark unknown positions as masked
                        pixel_mask[b, patch_idx, unknown_positions] = True
        
        # If no masking specified at all, use all pixels
        if mask_indices is None and partial_mask_indices is None:
            pixel_mask = torch.ones_like(pixel_mask)
        
        # Count masked pixels for validation
        num_masked_pixels = pixel_mask.sum().item()
        if num_masked_pixels == 0:
            # No pixels to predict, return zero loss
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Extract only the masked pixel positions
        masked_logits = logits[pixel_mask]
        masked_targets = original_patches[pixel_mask]
        
        # Calculate binary cross entropy loss on masked pixels only
        loss = F.binary_cross_entropy_with_logits(
            masked_logits,
            masked_targets,
            reduction='mean'
        )
        
        return loss
    
    def apply_masking(self, patches, num_masks, partial_mask_ratio=0.3):
        """
        Apply masking to patches with support for partial masking.
        Returns masking information but does NOT modify patches in-place.
        """
        batch_size, num_patches, patch_dim = patches.shape
        device = patches.device
        
        # Initialize mask tensors
        full_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        partial_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        
        # Make sure num_masks doesn't exceed the number of patches
        num_masks = min(num_masks, num_patches)
        
        # Calculate how many should be partial masks
        num_partial = min(int(num_masks * partial_mask_ratio), num_masks)
        num_full = num_masks - num_partial
        
        # Storage for partial mask values
        max_partial = max(1, num_partial)
        partial_values = torch.zeros((batch_size, max_partial, patch_dim), device=device)
        
        for b in range(batch_size):
            # Randomly select patches to mask
            mask_indices = torch.randperm(num_patches, device=device)[:num_masks]
            
            # Split between full and partial masks
            full_indices = mask_indices[:num_full]
            partial_indices = mask_indices[num_full:num_masks]
            
            # Mark fully masked patches
            full_mask[b, full_indices] = True
            
            # Handle partially masked patches
            partial_mask[b, partial_indices] = True
            
            # Create partial mask values
            for i, idx in enumerate(partial_indices):
                if i < max_partial:  # Safety check
                    # Get original patch (without modifying it)
                    orig_patch = patches[b, idx]
                    
                    # Create a partial mask
                    partial_patch = torch.ones(patch_dim, device=device) * 0.5
                    
                    # Randomly select positions to keep (1-3 positions)
                    num_to_keep = torch.randint(1, min(4, patch_dim), (1,)).item()
                    keep_positions = torch.randperm(patch_dim, device=device)[:num_to_keep]
                    
                    # Create a new tensor for the partial mask
                    for pos in keep_positions:
                        # New tensor creation instead of in-place modification
                        partial_patch[pos] = orig_patch[pos].clone()
                    
                    # Store the partial patch
                    partial_values[b, i] = partial_patch
        
        return {
            'full_mask': full_mask,
            'partial_mask': partial_mask,
            'partial_values': partial_values
        }
    
    def reconstruct_image(self, patch_values, image_size=64):
        """
        Reconstruct full image from patch values.
        """
        patches_per_dim = image_size // 4
        
        # Initialize output image
        image = torch.zeros((image_size, image_size), device=patch_values.device)
        
        # Place each patch in the correct position
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                # Get patch index
                idx = i * patches_per_dim + j
                if idx < patch_values.size(0):  # Safety check
                    # Reshape patch to 4x4
                    patch = patch_values[idx].reshape(4, 4)
                    
                    # Place patch in image (copy, not in-place)
                    image[i*4:(i+1)*4, j*4:(j+1)*4] = patch.clone()
        
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
        
        # Residual connection (addition, not in-place)
        z = z + self.dropout(attn_output)
        
        # Apply LayerNorm before feedforward
        z_norm = self.norm2(z)
        
        # Feedforward layer
        feedforward_output = self.feedforward(z_norm)
        
        # Final residual connection
        z = z + self.dropout(feedforward_output)
        
        return z
