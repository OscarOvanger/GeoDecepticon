import torch
from torch.utils.data import Dataset
import numpy as np

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        """
        Args:
            images (Tensor): Tensor of shape (num_images, 64, 64) with binary values (0 or 1).
        """
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        return torch.tensor(image, dtype=torch.float32)

class PatchTokenizer:
    """
    A class to handle tokenization of 4x4 patches using a dynamically created mapping.
    """
    def __init__(self, unique_patches, num_partial_masks=32):
        """
        Initialize the tokenizer with unique patches found in the training data.
        
        Args:
            unique_patches: List of unique patch tensors
            num_partial_masks: Number of partially masked tokens to support
        """
        self.unique_patches = unique_patches
        self.num_unique_patches = len(unique_patches)
        self.num_partial_masks = num_partial_masks
        
        # Create patch to token mappings
        self.patch_to_token = {}
        for i, patch in enumerate(unique_patches):
            self.patch_to_token[tuple(patch.tolist())] = i
        
        # Define special tokens
        self.mask_token = self.num_unique_patches
        self.partial_mask_start = self.mask_token + 1
        
        # Total number of tokens
        self.num_tokens = self.num_unique_patches + 1 + self.num_partial_masks
        
        # Create partial mask lookup table for efficiency
        self.partial_mask_lookup = {}
        partial_mask_idx = 0
        for position in range(min(16, num_partial_masks // 2)):
            for value in [0, 1]:
                if partial_mask_idx >= num_partial_masks:
                    break
                key = (position, value)
                self.partial_mask_lookup[key] = self.partial_mask_start + partial_mask_idx
                partial_mask_idx += 1
    
    def encode(self, patch):
        """
        Convert a 4x4 patch to a token ID.
        
        Args:
            patch: Tensor of shape (16) representing a 4x4 binary patch
            
        Returns:
            int: Token ID for the patch
        """
        # Check if the patch contains masked values (0.5 or -1)
        contains_masked = torch.any((patch == 0.5) | (patch == -1))
        
        if contains_masked:
            # Replace any -1 values with 0.5 for consistency
            patch = torch.where(patch == -1, torch.tensor(0.5), patch)
            
            # If fully masked, return the mask token
            if torch.all(patch == 0.5):
                return self.mask_token
            
            # If partially masked with exactly one known value
            known_positions = torch.where(patch != 0.5)[0]
            if len(known_positions) == 1:
                position = known_positions[0].item()
                value = int(patch[position].item())
                
                # Check if this partial mask configuration is supported
                key = (position, value)
                if key in self.partial_mask_lookup:
                    return self.partial_mask_lookup[key]
            
            # Default to mask token for unsupported partial mask configurations
            return self.mask_token
        
        # For standard binary patches, look up in the mapping
        patch_tuple = tuple(patch.tolist())
        if patch_tuple in self.patch_to_token:
            return self.patch_to_token[patch_tuple]
        else:
            # If we encounter a patch not in training data, use mask token
            print(f"Warning: Encountered unknown patch pattern: {patch_tuple}")
            return self.mask_token
    
    def decode(self, token_id):
        """
        Convert a token ID back to a 4x4 patch.
        
        Args:
            token_id: Integer token ID
            
        Returns:
            Tensor: 4x4 patch values
        """
        if token_id < self.num_unique_patches:
            # Return the corresponding unique patch
            return self.unique_patches[token_id]
        
        elif token_id == self.mask_token:
            # Return fully masked patch
            return torch.tensor([0.5] * 16)
        
        elif token_id >= self.partial_mask_start and token_id < self.num_tokens:
            # Handle partially masked token
            partial_mask_idx = token_id - self.partial_mask_start
            position = partial_mask_idx // 2
            value = partial_mask_idx % 2
            
            # Create a fully masked patch with one known value
            patch = torch.tensor([0.5] * 16)
            patch[position] = float(value)
            return patch
        
        else:
            raise ValueError(f"Invalid token ID: {token_id}")


def preprocess_image(image, tokenizer):
    """
    Splits a binary 64x64 image into flattened 4x4 patches and converts to token IDs.
    
    Args:
        image: Tensor of shape (64, 64) with binary values
        tokenizer: PatchTokenizer instance for encoding patches
        
    Returns:
        Tensor: Token IDs for each 4x4 patch in the image (256 tokens for 64x64 image)
    """
    # Ensure image is the right shape
    if image.shape != (64, 64):
        raise ValueError(f"Expected image shape (64, 64), got {image.shape}")
    
    # Extract 4x4 patches
    patches = image.unfold(0, 4, 4).unfold(1, 4, 4)
    patches = patches.contiguous().view(-1, 16)  # Shape: (256, 16)
    
    # Convert each patch to a token ID
    patch_indices = torch.zeros(patches.size(0), dtype=torch.long)
    for i, patch in enumerate(patches):
        patch_indices[i] = tokenizer.encode(patch)
    
    return patch_indices


def reconstruct_image_from_patches(patch_indices, tokenizer):
    """
    Reconstructs a 64x64 image from patch token indices.
    
    Args:
        patch_indices: Tensor of token indices of shape (256)
        tokenizer: PatchTokenizer instance for decoding tokens
        
    Returns:
        Tensor: Reconstructed image of shape (64, 64)
    """
    # Initialize the reconstructed image
    reconstructed_image = torch.zeros((64, 64))
    
    # Calculate the number of patches in each dimension
    patches_per_dim = 16  # 16 patches per dimension for 4x4 patches in a 64x64 image
    
    for i in range(patches_per_dim):
        for j in range(patches_per_dim):
            # Get the patch index
            idx = i * patches_per_dim + j
            token_id = patch_indices[idx].item()
            
            # Decode the token to get patch values
            patch_values = tokenizer.decode(token_id)
            
            # Reshape to 4x4
            patch_values = patch_values.reshape(4, 4)
            
            # Place the patch in the image
            reconstructed_image[i*4:(i+1)*4, j*4:(j+1)*4] = patch_values
    
    return reconstructed_image
