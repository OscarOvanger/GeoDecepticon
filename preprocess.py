import torch
from torch.utils.data import Dataset
import numpy as np

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

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        """
        Args:
            images (np.ndarray or Tensor): Array of shape (num_images, height, width) with binary values
        """
        # Ensure images are torch tensors
        if isinstance(images, np.ndarray):
            self.images = torch.FloatTensor(images)
        else:
            self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        return image

def extract_patches(image, patch_size=4):
    """
    Extract patches from image.
    
    Args:
        image: Tensor of shape [height, width]
        patch_size: Size of the patches
        
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
    Collate function for dataloader that extracts patches.
    
    Args:
        batch: List of images
    
    Returns:
        Tensor of shape [batch_size, num_patches, patch_dim]
    """
    batch_patches = []
    for image in batch:
        patches = extract_patches(image)
        batch_patches.append(patches)
    
    return torch.stack(batch_patches)

def calculate_mask_count(epoch, num_epochs, max_patches=256):
    """
    Sigmoid-based masking schedule to gradually approach 100% masking.
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        max_patches: Maximum number of patches to mask (usually total patches)
        
    Returns:
        int: Number of patches to mask
    """
    min_masks = 4
    # Use sigmoid function for smoother progression to 100% masking
    # Slower at start and end, faster in middle
    progress = epoch / num_epochs
    # Steepness of 10 centers the faster growth around the middle epochs
    sigmoid = 1 / (1 + np.exp(-10 * (progress - 0.5)))
    # Scale sigmoid output (0-1) to range from min_masks to max_patches
    mask_count = min_masks + (max_patches - min_masks) * sigmoid
    return int(mask_count)

def get_lr(epoch, num_epochs, base_lr=1e-4, min_lr=1e-6):
    """
    Learning rate schedule with warmup and cosine decay.
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        float: Learning rate for current epoch
    """
    warmup = 5  # Warmup epochs
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    else:
        # Cosine decay with minimum lr
        progress = (epoch - warmup) / max(1, (num_epochs - warmup))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

def generate_complete_image(model, device='cuda', batch_size=16, size=64):
    """
    Generate complete images from scratch (100% masked reconstruction).
    
    Args:
        model: Trained ContinuousVisionTransformer model
        device: Device to use
        batch_size: Number of images to generate
        size: Image size (default 64x64)
        
    Returns:
        torch.Tensor: Generated images
    """
    model.eval()
    patch_size = 4
    num_patches = (size // patch_size) ** 2  # 256 for 64x64 image with 4x4 patches
    patch_dim = 16  # 4x4 patches flattened
    
    # Create dummy patches (all zeros)
    patches = torch.zeros((batch_size, num_patches, patch_dim), device=device)
    
    # Create mask for all patches (100% masking)
    mask = torch.ones((batch_size, num_patches), dtype=torch.bool, device=device)
    
    # No partial masking in this case
    partial_mask = torch.zeros((batch_size, num_patches), dtype=torch.bool, device=device)
    
    # Generate images
    with torch.no_grad():
        outputs = model(
            patches, 
            mask_indices=mask, 
            partial_mask_indices=partial_mask,
            partial_mask_values=None
        )
        
    # Get binary predictions
    binary_pred = outputs['binary_prediction']
    
    # Reconstruct full images
    generated_images = []
    for i in range(batch_size):
        image = model.reconstruct_image(binary_pred[i])
        generated_images.append(image)
    
    return torch.stack(generated_images)

def visualize_reconstructions(model, dataloader, device, epoch, num_masks=None):
    """
    Create visualizations for original, masked, and reconstructed images.
    Also generate fully masked (100%) reconstructions.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device to use
        epoch: Current epoch
        num_masks: Number of masks to apply (optional)
    
    Returns:
        dict: Dictionary of images for wandb logging
    """
    model.eval()
    
    # Get a batch of data
    patches = next(iter(dataloader)).to(device)
    batch_size = min(1, patches.size(0))  # Limit to 4 images for visualization
    patches = patches[:batch_size]
    
    # Apply normal masking
    if num_masks is None:
        num_masks = 64  # Default moderate masking
    
    masking_info = model.apply_masking(
        patches, 
        num_masks=num_masks,
        partial_mask_ratio=0.3
    )
    
    # Get reconstructions
    with torch.no_grad():
        outputs = model(
            patches,
            mask_indices=masking_info['full_mask'],
            partial_mask_indices=masking_info['partial_mask'],
            partial_mask_values=masking_info['partial_values']
        )
    
    binary_pred = outputs['binary_prediction']
    
    # Create visualizations
    images = {}
    
    for i in range(batch_size):
        # Original image
        original = model.reconstruct_image(patches[i]).cpu()
        
        # Create masked visualization
        masked_patches = patches[i].clone()
        full_mask = masking_info['full_mask'][i]
        partial_mask = masking_info['partial_mask'][i]
        
        # Set fully masked patches to 0.5 (gray)
        for idx in torch.where(full_mask)[0]:
            masked_patches[idx] = torch.ones_like(masked_patches[idx]) * 0.5
        
        # For partially masked patches, use the partial values
        for j, idx in enumerate(torch.where(partial_mask)[0]):
            if j < masking_info['partial_values'].size(1):
                masked_patches[idx] = masking_info['partial_values'][i, j]
        
        masked = model.reconstruct_image(masked_patches).cpu()
        
        # Reconstructed image
        reconstructed = model.reconstruct_image(binary_pred[i]).cpu()
        
        # Add to dictionary
        images[f"original_{i}"] = wandb.Image(original, caption=f"Original {i}")
        images[f"masked_{i}"] = wandb.Image(masked, caption=f"Masked {i} ({num_masks} masks)")
        images[f"reconstructed_{i}"] = wandb.Image(reconstructed, caption=f"Reconstructed {i}")
    
    # Generate 100% masked images
    fully_generated = generate_complete_image(model, device, batch_size=batch_size)
    for i in range(batch_size):
        images[f"generated_{i}"] = wandb.Image(fully_generated[i].cpu(), caption=f"100% Masked Generation {i}")
    
    return images
