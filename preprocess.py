import torch
from torch.utils.data import Dataset

class BinaryImageDataset(Dataset):
    """
    Dataset class for binary images.
    """
    def __init__(self, images):
        """
        Args:
            images (numpy array or tensor): Array of shape (num_images, height, width) with binary values
        """
        # Ensure images are torch tensors
        if not isinstance(images, torch.Tensor):
            self.images = torch.FloatTensor(images)
        else:
            self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]

def extract_patches(image, patch_size=4):
    """
    Extract patches from a single image.
    
    Args:
        image: Tensor of shape [height, width]
        patch_size: Size of each patch
        
    Returns:
        Tensor of shape [num_patches, patch_size*patch_size]
    """
    # Ensure image is the right shape
    if len(image.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    # Extract patches
    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size*patch_size)
    
    return patches

def collate_fn(batch):
    """
    Collate function for DataLoader to convert images to patches.
    
    Args:
        batch: List of images from the dataset
    
    Returns:
        Tensor of shape [batch_size, num_patches, patch_dim]
    """
    batch_patches = []
    for image in batch:
        patches = extract_patches(image)
        batch_patches.append(patches)
    
    return torch.stack(batch_patches)

def create_masking_info(patches, num_masks, partial_mask_ratio=0.3):
    """
    Create masking information for training.
    
    Args:
        patches: Tensor of shape [batch_size, num_patches, patch_dim]
        num_masks: Number of patches to mask
        partial_mask_ratio: Ratio of partial masks vs. full masks
        
    Returns:
        dict: Contains 'full_mask', 'partial_mask', and 'partial_values'
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
    
    # Storage for partial mask values (pre-allocate max size)
    max_partial = max(1, num_partial)  # Ensure at least size 1 to avoid empty tensor issues
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
            # Get original patch
            orig_patch = patches[b, idx].clone()
            
            # Create a partial mask where most values are masked (0.5)
            # but one or a few values are kept
            partial_patch = torch.ones_like(orig_patch) * 0.5
            
            # Randomly select positions to keep (1-3 positions)
            num_to_keep = torch.randint(1, min(4, patch_dim), (1,)).item()
            keep_positions = torch.randperm(patch_dim)[:num_to_keep]
            
            # Keep original values at selected positions
            for pos in keep_positions:
                partial_patch[pos] = orig_patch[pos]
            
            # Store partial patch values
            if i < partial_values.size(1):  # Safety check
                partial_values[b, i] = partial_patch
    
    return {
        'full_mask': full_mask,
        'partial_mask': partial_mask,
        'partial_values': partial_values
    }
