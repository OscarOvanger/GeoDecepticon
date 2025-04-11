import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#from tqdm import tqdm  # you can uncomment if desired
import numpy as np
import matplotlib.pyplot as plt
import random
import time

########################################
# Helper: Convert Patches to Integer Code
########################################

def patch_to_int(patches):
    """
    Converts a tensor of patches (shape: [N, patch_dim]) containing binary values
    to an integer code. Assumes values are near 0 or 1.
    """
    patches_bin = patches.round().long()  # ensure binary
    patch_dim = patches_bin.shape[1]
    powers = (2 ** torch.arange(patch_dim - 1, -1, -1, device=patches.device)).unsqueeze(0)
    codes = (patches_bin * powers).sum(dim=1)
    return codes

def training_dat_to_patches(training_data, patch_size):
    """Extract patches from training data"""
    N, H, W = training_data.shape
    # Use unfold to extract non-overlapping patches
    x = training_data.reshape(N, 1, H, W)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).reshape(-1, patch_size * patch_size)

########################################
# Build enhanced vocabulary with learnable tokens
########################################

def build_vocabulary(training_data, patch_size, partial_masked_token=True):
    """Build vocabulary of discrete tokens including masked tokens"""
    # Extract patches
    patches = training_dat_to_patches(training_data, patch_size)
    
    # Binary patterns - include all possible binary configurations
    vocab = []
    for i in range(2**(patch_size**2)):
        binary = torch.tensor([int(bit) for bit in np.binary_repr(i, width=patch_size**2)], dtype=torch.float32)
        vocab.append(binary)
    
    # Add fully masked token
    fully_masked_token = torch.ones(patch_size**2) * 0.5
    vocab.append(fully_masked_token)
    fully_masked_idx = len(vocab) - 1
    
    # Add partially masked tokens
    partial_token_indices = []
    if partial_masked_token:
        for pos in range(patch_size**2):
            for val in range(2):
                token = torch.ones(patch_size**2) * 0.5
                token[pos] = float(val)
                vocab.append(token)
                partial_token_indices.append(len(vocab) - 1)
    
    # Create vocabulary tensor
    vocab = torch.stack(vocab)
    
    # Create trainable masked token
    masked_token_param = nn.Parameter(torch.mean(patches.float(), dim=0), requires_grad=True)
    
    # Create trainable partial masked tokens
    partial_masked_token_params = []
    if partial_masked_token:
        for pos in range(patch_size**2):
            for val in range(2):
                # Find relevant patches
                mask = patches[:, pos] == val
                if mask.any():
                    mean_patch = torch.mean(patches[mask].float(), dim=0)
                    param = nn.Parameter(mean_patch.clone(), requires_grad=True)
                else:
                    param = nn.Parameter(fully_masked_token.clone(), requires_grad=True)
                    param[pos] = float(val)
                
                partial_masked_token_params.append(param)
    
    return vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params

########################################
# Relative Position Self-Attention (Modified)
########################################

class RelativePositionSelfAttention(nn.Module):
    """Self-attention with learnable relative positional encoding.
       Here we create a parameter of shape (max_seq_len, max_seq_len) which is sliced
       to the current sequence length and passed as an attn_mask (added to Q*K^T)."""
    def __init__(self, hidden_dim, num_heads, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.max_seq_len = max_seq_len
        # Create a learnable matrix of shape (max_seq_len, max_seq_len)
        self.rel_pos_encoding = nn.Parameter(torch.zeros(max_seq_len, max_seq_len))
        nn.init.normal_(self.rel_pos_encoding, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        # Slice the relative positional encoding to current sequence length.
        attn_mask = self.rel_pos_encoding[:seq_len, :seq_len]
        # Pass attn_mask to the MHA layer (it will be added to Q*K^T)
        output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        return output

########################################
# Transformer Encoder Layer
########################################

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with relative positional encoding"""
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1, max_seq_len=256):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RelativePositionSelfAttention(hidden_dim, num_heads, max_seq_len, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

########################################
# Vision Transformer
########################################

class VisionTransformer(nn.Module):
    """Vision Transformer with discrete token vocabulary and relative positional encoding"""
    def __init__(self, vocab, fully_masked_idx, partial_token_indices, 
                 masked_token_param, partial_masked_token_params, patch_size, 
                 num_heads, num_layers, ffn_dim, hidden_dim, max_seq_len=256, dropout=0.1):
        super().__init__()
        # Store vocabulary and token parameters
        self.register_buffer('vocab', vocab)
        self.fully_masked_idx = fully_masked_idx
        self.partial_token_indices = partial_token_indices  # list; will be converted as needed
        self.masked_token_param = masked_token_param  # trainable masked token
        self.partial_masked_token_params = nn.ParameterList(partial_masked_token_params)
        self.mask_token = self.masked_token_param  # For compatibility
        
        self.patch_size = patch_size
        self.vocab_size = vocab.shape[0]
        
        # Precompute vocabulary integer codes (for any later fast lookups)
        self.vocab_int = patch_to_int(self.vocab)
        
        # Embedding layer for patches
        self.token_embedding = nn.Linear(patch_size**2, hidden_dim)
        
        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, max_seq_len)
            for _ in range(num_layers)
        ])
        
        # Output projection layer
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.zeros_(self.token_embedding.bias)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def _process_batch(self, patches):
        """
        Vectorized patch processing. For each patch in a batch, determine if it is:
          - Fully masked: assign masked token.
          - Partially masked: find the first binary (unmasked) position and use that to index into partial tokens.
          - Complete binary patch: compute nearest vocabulary token by Euclidean distance.
        """
        B, num_patches, patch_dim = patches.shape
        device = patches.device
        N = B * num_patches
        
        patches_flat = patches.view(N, patch_dim)  # shape: (N, patch_dim)
        
        # Fully masked if all entries are near 0.5
        is_fully_masked = torch.all(torch.abs(patches_flat - 0.5) < 0.1, dim=1)  # (N,)
        
        # For each patch, compute which entries are “masked” (close to 0.5)
        mask_positions = torch.abs(patches_flat - 0.5) < 0.1  # (N, patch_dim)
        binary_positions = ~mask_positions
        has_mask = mask_positions.any(dim=1)
        has_binary = binary_positions.any(dim=1)
        is_partial = (~is_fully_masked) & (has_mask & has_binary)
        is_complete = (~is_fully_masked) & (~is_partial)
        
        # Prepare output tensors
        token_indices_flat = torch.empty((N,), dtype=torch.long, device=device)
        processed_patches_flat = torch.empty_like(patches_flat)
        
        # --- Fully masked patches ---
        fully_masked_idxes = torch.nonzero(is_fully_masked).squeeze(1)
        if fully_masked_idxes.numel() > 0:
            token_indices_flat[fully_masked_idxes] = self.fully_masked_idx
            masked_token = self.masked_token_param.unsqueeze(0).expand(len(fully_masked_idxes), patch_dim)
            processed_patches_flat[fully_masked_idxes] = masked_token
        
        # --- Partial patches ---
        partial_idxes = torch.nonzero(is_partial).squeeze(1)
        if partial_idxes.numel() > 0:
            # For each partial patch, find the first binary (unmasked) position.
            binary_vals = binary_positions[partial_idxes].to(torch.int64)  # (n_partial, patch_dim)
            first_pos = torch.argmax(binary_vals, dim=1)  # first index where binary_positions is True
            # Gather the actual value at that position and round it (expected 0 or 1)
            chosen_val = torch.round(patches_flat[partial_idxes].gather(1, first_pos.unsqueeze(1))).squeeze(1)
            # Compute index: partial token index = position * 2 + value
            partial_token_idx = first_pos * 2 + chosen_val.to(torch.long)
            # Convert stored list of partial token vocabulary indices to a tensor (if not done already)
            if not hasattr(self, 'partial_token_indices_tensor'):
                self.partial_token_indices_tensor = torch.tensor(self.partial_token_indices, device=device)
            selected_vocab_idx = self.partial_token_indices_tensor[partial_token_idx]
            token_indices_flat[partial_idxes] = selected_vocab_idx
            # Stack parameters from the ParameterList if not already done
            if not hasattr(self, 'partial_token_params_tensor'):
                self.partial_token_params_tensor = torch.stack(list(self.partial_masked_token_params), dim=0)
            processed_patches_flat[partial_idxes] = self.partial_token_params_tensor[partial_token_idx]
        
        # --- Complete binary patches ---
        complete_idxes = torch.nonzero(is_complete).squeeze(1)
        if complete_idxes.numel() > 0:
            complete_patches = patches_flat[complete_idxes]
            # Compute Euclidean distances to all vocabulary tokens in a vectorized manner.
            dists = torch.cdist(complete_patches, self.vocab)
            closest = torch.argmin(dists, dim=1)
            token_indices_flat[complete_idxes] = closest
            # For complete patches, use the original patch values.
            processed_patches_flat[complete_idxes] = complete_patches
        
        # Reshape back to original batch dimensions.
        token_indices = token_indices_flat.view(B, num_patches)
        processed_patches = processed_patches_flat.view(B, num_patches, patch_dim)
        
        return processed_patches, token_indices
    
    def forward(self, patches):
        # Ensure patches have batch dimension
        if patches.dim() == 2:
            patches = patches.unsqueeze(0)
            
        processed_patches, token_indices = self._process_batch(patches)
        
        # Generate patch embeddings
        embeddings = self.token_embedding(processed_patches)
        embeddings = self.dropout(embeddings)
        
        # Transformer pass
        z = self.transformer(embeddings)
        
        # Project into vocabulary space
        logits = self.output_projection(z)
        
        return logits
    
    def _find_closest_token(self, patch):
        """
        Fallback function that computes the nearest token in the vocabulary for a given patch.
        (This method is retained for compatibility but is now replaced by vectorized computations.)
        """
        distances = torch.sum((self.vocab - patch.unsqueeze(0))**2, dim=1)
        return torch.argmin(distances)

########################################
# Dataset and Utility Functions
########################################

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        self.images = images  # shape: (N, H, W)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
    @staticmethod
    def batch_to_patches(images, patch_size):
        # Vectorized extraction using unfold.
        B, H, W = images.shape
        patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)
        return patches
    
    @staticmethod
    def patches_to_image(patches, patch_size, image_size):
        # Vectorized conversion from patches to image.
        n_patches = patches.shape[0]
        grid_size = image_size // patch_size
        patches_grid = patches.view(grid_size, grid_size, patch_size, patch_size)
        image = patches_grid.permute(0,2,1,3).reshape(image_size, image_size)
        return image

########################################
# Reconstruction Function (Vectorized)
########################################

def test_reconstruction(model, original_sample, masked_sample):
    """
    Reconstruct masked patches in an image using vectorized patch extraction and merging.
    Only patches that are masked (i.e. contain values close to 0.5) are replaced by the model prediction.
    """
    H, W = original_sample.shape
    patch_size = model.patch_size
    n_h, n_w = H // patch_size, W // patch_size
    
    # Extract patches vectorized.
    patches = masked_sample.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size * patch_size)  # (n_patches, patch_dim)
    
    # Identify masked patches.
    mask_flags = torch.any(torch.abs(patches - 0.5) < 0.1, dim=1)
    
    # Get model predictions for all patches.
    patches_input = patches.unsqueeze(0)  # shape (1, n_patches, patch_dim)
    with torch.no_grad():
        logits = model(patches_input)
    token_indices = torch.argmax(logits, dim=-1).squeeze(0)  # (n_patches)
    predictions = model.vocab[token_indices]  # (n_patches, patch_dim)
    
    # Replace only masked patches.
    patches[mask_flags] = predictions[mask_flags]
    
    # Reconstruct image using vectorized reordering.
    reconstructed = patches.view(n_h, n_w, patch_size, patch_size)\
                           .permute(0, 2, 1, 3)\
                           .reshape(H, W)
    return reconstructed

########################################
# Test Script for Isolating Inner Workings
########################################

if __name__ == '__main__':
    # Set a random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create a dummy binary image (e.g., 12x12 image for patch_size=3 -> 16 patches)
    H, W = 12, 12
    patch_size = 3
    # Create a simple pattern (for clarity, you can also use random)
    dummy_image = np.random.randint(0, 2, (H, W)).astype(np.float32)
    print("Original Image:\n", dummy_image)
    
    # Convert to torch tensor
    dummy_image_tensor = torch.tensor(dummy_image)
    dummy_image_tensor = dummy_image_tensor.unsqueeze(0)  # shape: (1, H, W)
    
    # Build vocabulary from a dummy dataset (using this image alone)
    vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params = build_vocabulary(
        dummy_image_tensor, patch_size, partial_masked_token=True
    )
    
    # Create a VisionTransformer (using relatively small parameters for debugging)
    hidden_dim = 64
    num_heads = 4
    num_layers = 2
    ffn_dim = 128
    max_seq_len = (H // patch_size) * (W // patch_size)  # here, 16 patches
    model = VisionTransformer(
        vocab=vocab,
        fully_masked_idx=fully_masked_idx,
        partial_token_indices=partial_token_indices,
        masked_token_param=masked_token_param,
        partial_masked_token_params=partial_masked_token_params,
        patch_size=patch_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        hidden_dim=hidden_dim,
        max_seq_len=max_seq_len,
        dropout=0.1
    )
    
    # Move model to device (CPU for testing)
    device = torch.device("cpu")
    model.to(device)
    
    # Extract patches before masking
    patches = BinaryImageDataset.batch_to_patches(dummy_image_tensor, patch_size)
    print("\nPatches before masking (shape {}):".format(patches.shape))
    print(patches)
    
    # Simulate a masking step.
    # For simplicity, let’s create a mask that marks half the patches as masked.
    B, num_patches, patch_dim = patches.shape
    mask_ratio = 0.5
    mask = torch.rand(B, num_patches) < mask_ratio
    
    # Create a copy of patches for masked_patches.
    masked_patches = patches.clone()
    # For fully masked patches: replace with the model's mask token (value 0.5 vector)
    masked_patches[mask] = model.mask_token  # full masking; ignore partial for now
    print("\nPatches after masking:")
    print(masked_patches)
    
    # Check the output of _process_batch:
    processed_patches, token_indices = model._process_batch(masked_patches)
    print("\nProcessed Patches after _process_batch (should have masked tokens for masked patches):")
    print(processed_patches)
    print("\nToken Indices from _process_batch:")
    print(token_indices)
    
    # Forward pass: get logits
    logits = model(masked_patches)
    print("\nLogits shape (should be [B, num_patches, vocab_size]):", logits.shape)
    # Print logits for first patch as an example
    print("\nLogits for first patch:")
    print(logits[0, 0])
    
    # Check _find_closest_token on the first original patch:
    sample_patch = patches[0,0]
    closest_token_idx = model._find_closest_token(sample_patch)
    print("\nFirst patch (original):")
    print(sample_patch)
    print("\nClosest token index (via _find_closest_token):", closest_token_idx.item())
    print("Vocabulary entry for that token:")
    print(model.vocab[closest_token_idx])
    
    # Optionally, compute a loss on the masked patches.
    # Compute vectorized target tokens (for all patches in the batch)
    patches_flat = patches.view(B * num_patches, patch_dim)
    dists = torch.cdist(patches_flat, model.vocab)
    targets = torch.argmin(dists, dim=1).view(B, num_patches)
    criterion = nn.CrossEntropyLoss()
    if mask.sum() > 0:
        loss = criterion(logits[mask], targets[mask])
        print("\nLoss computed on masked patches:", loss.item())
    else:
        print("\nNo patches masked; cannot compute loss.")
    
    # For visualization, you can also reconstruct the image.
    # Simulate a masked image: (set masked patches to 0.5 again)
    dummy_masked_image = dummy_image_tensor.clone()
    # For each patch in the image, if its corresponding patch in masked_patches is mask_token, set it to 0.5.
    # (This is a simplistic reassembly for visualization.)
    reconstructed = test_reconstruction(model, dummy_image_tensor.squeeze(0), dummy_masked_image.squeeze(0))
    print("\nReconstructed image (as tensor):")
    print(reconstructed)
    
    # Optionally visualize using matplotlib:
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(dummy_image, cmap='gray')
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(masked_patches[0].view(H // patch_size, W // patch_size, patch_size, patch_size)
               .permute(0,2,1,3).reshape(H, W).cpu().numpy(), cmap='gray')
    plt.title("Masked Patches (Reassembled)")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed.cpu().numpy(), cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
