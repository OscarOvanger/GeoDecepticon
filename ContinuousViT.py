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
        # This will be added to patches that are partially masked
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
            mask_indices: Boolean tensor of shape [batch_size, num_patches] indicating fully masked patches
            partial_mask_indices: Boolean tensor of shape [batch_size, num_patches] indicating partially masked patches
            partial_mask_values: Tensor of shape [batch_size, num_partial_masks, 16] with values for partial masks
                                 (0.5 indicates masked positions, 0 or 1 indicates known positions)
        
        Returns:
            outputs: Dictionary containing model outputs
        """
        batch_size, num_patches, patch_dim = patches.shape
        device = patches.device
        
        # Create patch embeddings through linear projection
        patch_embeddings = self.patch_projection(patches)  # [batch, num_patches, embed_dim]
        
        # Store original patches for loss calculation
        original_patches = patches.clone()
        
        # Apply masking if specified
        if mask_indices is not None:
            # Handle fully masked patches
            for b in range(batch_size):
                # Get indices of fully masked patches for this batch
                full_mask_idx = torch.where(mask_indices[b])[0]
                if len(full_mask_idx) > 0:
                    # Replace with mask token
                    patch_embeddings[b, full_mask_idx] = self.mask_token.expand(len(full_mask_idx), -1)
        
        # Apply partial masking if specified
        if partial_mask_indices is not None and partial_mask_values is not None:
            for b in range(batch_size):
                # Get indices of partially masked patches for this batch
                partial_mask_idx = torch.where(partial_mask_indices[b])[0]
                if len(partial_mask_idx) > 0:
                    # Replace patch values with partial mask values
                    for i, idx in enumerate(partial_mask_idx):
                        if i < partial_mask_values.size(1):  # Safety check
                            # Get the partial mask pattern
                            partial_values = partial_mask_values[b, i]
                            
                            # Create partial patch: copy original values but mask some positions
                            partial_patch = patches[b, idx].clone()
                            mask_positions = (partial_values == 0.5)
                            
                            # Zero out masked positions in the patch
                            # (but keep the known values from partial_values)
                            partial_patch = torch.where(
                                mask_positions,
                                torch.zeros_like(partial_patch),
                                partial_values
                            )
                            
                            # Replace the patch with partial mask
                            patches[b, idx] = partial_patch
                            
                            # Update embedding with new value and add partial mask indicator
                            patch_embeddings[b, idx] = self.patch_projection(partial_patch) + self.partial_mask_indicator
        
        # Add position embeddings
        embeddings = patch_embeddings + self.pos_embedding[:, :num_patches]  # [batch, num_patches, embed_dim]
        
        # Process through transformer layers
        x = embeddings.permute(1, 0, 2)  # [num_patches, batch, embed_dim]
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.permute(1, 0, 2)  # [batch, num_patches, embed_dim]
        
        # Decode to patch values
        patch_logits = self.patch_decoder(x)  # [batch, num_patches, patch_dim]
        
        # Apply sigmoid to get probabilities for each position
        patch_probs = torch.sigmoid(patch_logits)
        
        # For inference, round probabilities to binary values
        binary_pred = (patch_probs > 0.5).float()
        
        # Prepare outputs
        outputs = {
            'logits': patch_logits,         # Raw logits for each position
            'probabilities': patch_probs,   # Probability of each position being 1
            'binary_prediction': binary_pred,  # Thresholded binary prediction
            'embeddings': x                 # Final embeddings
        }
        
        return outputs
    
    def get_loss(self, outputs, original_patches, mask_indices=None, partial_mask_indices=None):
        """
        Calculate BCE loss for patch reconstruction.
        
        Args:
            outputs: Dictionary from forward pass
            original_patches: Original patch values [batch, num_patches, patch_dim]
            mask_indices: Boolean tensor for fully masked patches [batch, num_patches]
            partial_mask_indices: Boolean tensor for partially masked patches [batch, num_patches]
            
        Returns:
            loss: Binary cross entropy loss
        """
        batch_size, num_patches, patch_dim = original_patches.shape
        logits = outputs['logits']
        
        # Combine full and partial mask indices
        if mask_indices is not None and partial_mask_indices is not None:
            combined_mask = mask_indices | partial_mask_indices
        elif mask_indices is not None:
            combined_mask = mask_indices
        elif partial_mask_indices is not None:
            combined_mask = partial_mask_indices
        else:
            # If no masks specified, compute loss over all patches
            combined_mask = torch.ones_like(original_patches[:, :, 0], dtype=torch.bool)
        
        # Flatten tensors for loss calculation
        masked_logits = []
        masked_targets = []
        
        for b in range(batch_size):
            # Get indices of masked patches for this batch
            masked_idx = torch.where(combined_mask[b])[0]
            if len(masked_idx) > 0:
                # Extract logits and targets for masked positions
                b_logits = logits[b, masked_idx].reshape(-1)  # Flatten
                b_targets = original_patches[b, masked_idx].reshape(-1)  # Flatten
                
                masked_logits.append(b_logits)
                masked_targets.append(b_targets)
        
        if masked_logits:
            # Concatenate all masked positions
            masked_logits = torch.cat(masked_logits)
            masked_targets = torch.cat(masked_targets)
            
            # Calculate binary cross entropy loss with logits
            loss = F.binary_cross_entropy_with_logits(
                masked_logits, 
                masked_targets,
                reduction='mean'
            )
        else:
            # Default loss if no masked positions
            loss = torch.tensor(0.0, device=logits.device)
        
        return loss
    
    def apply_masking(self, patches, num_masks, partial_mask_ratio=0.3):
        """
        Apply masking to patches with support for partial masking.
        
        Args:
            patches: Tensor of shape [batch_size, num_patches, patch_dim]
            num_masks: Number of patches to mask per image
            partial_mask_ratio: Ratio of masks that should be partial
            
        Returns:
            dict: Contains masked patches and masking information
        """
        batch_size, num_patches, patch_dim = patches.shape
        device = patches.device
        
        # Initialize mask tensors
        full_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        partial_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
        
        # Storage for partial mask values (pre-allocate max size)
        max_partial = int(num_masks * partial_mask_ratio) + 1
        partial_values = torch.zeros((batch_size, max_partial, patch_dim), device=device)
        
        for b in range(batch_size):
            # Randomly select patches to mask
            mask_indices = torch.randperm(num_patches, device=device)[:num_masks]
            
            # Split between full and partial masks
            num_partial = int(num_masks * partial_mask_ratio)
            full_indices = mask_indices[num_partial:]
            partial_indices = mask_indices[:num_partial]
            
            # Mark fully masked patches
            full_mask[b, full_indices] = True
            
            # Handle partially masked patches
            partial_mask[b, partial_indices] = True
            
            # Create partial mask values
            for i, idx in enumerate(partial_indices):
                # Get original patch
                orig_patch = patches[b, idx].clone()
                
                # Create a partial mask where most values are masked (0.5)
                # but one or a few values are kept
                partial_patch = torch.ones_like(orig_patch) * 0.5
                
                # Randomly select positions to keep (1-3 positions)
                num_to_keep = torch.randint(1, 4, (1,)).item()
                keep_positions = torch.randperm(patch_dim)[:num_to_keep]
                
                # Keep original values at selected positions
                for pos in keep_positions:
                    partial_patch[pos] = orig_patch[pos]
                
                # Store partial patch values
                partial_values[b, i] = partial_patch
        
        return {
            'full_mask': full_mask,
            'partial_mask': partial_mask,
            'partial_values': partial_values
        }
    
    def reconstruct_image(self, patch_values, image_size=64):
        """
        Reconstruct full image from patch values.
        
        Args:
            patch_values: Tensor of shape [num_patches, patch_dim] with binary values
            image_size: Size of the square output image
            
        Returns:
            Tensor: Reconstructed image of shape [image_size, image_size]
        """
        # For 4x4 patches in a 64x64 image
        patches_per_dim = image_size // 4
        
        # Initialize output image
        image = torch.zeros((image_size, image_size), device=patch_values.device)
        
        # Place each patch in the correct position
        for i in range(patches_per_dim):
            for j in range(patches_per_dim):
                # Get patch index
                idx = i * patches_per_dim + j
                
                # Reshape patch to 4x4
                patch = patch_values[idx].reshape(4, 4)
                
                # Place patch in image
                image[i*4:(i+1)*4, j*4:(j+1)*4] = patch
        
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
            nn.GELU(),  # Using GELU as in standard ViT
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
