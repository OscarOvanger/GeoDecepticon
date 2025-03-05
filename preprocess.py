import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb
from generating_images import generate_image_autoregressively

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
    Improved sigmoid-based masking schedule for faster progression to 100% masking.
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        max_patches: Maximum number of patches to mask
        
    Returns:
        int: Number of patches to mask
    """
    min_masks = 4
    # Steeper sigmoid curve with midpoint earlier in training (40% of epochs)
    # Slope increased from 10 to 12 for faster transition
    progress = epoch / num_epochs
    sigmoid = 1 / (1 + np.exp(-12 * (progress - 0.4)))
    mask_count = min_masks + (max_patches - min_masks) * sigmoid
    return int(mask_count)

def get_lr(epoch, num_epochs, base_lr=1e-4, min_lr=1e-6):
    """
    Learning rate schedule with warmup, milestone boosts, and cosine decay.
    
    Args:
        epoch: Current epoch
        num_epochs: Total number of epochs
        base_lr: Base learning rate
        min_lr: Minimum learning rate
        
    Returns:
        float: Learning rate for current epoch
    """
    warmup = 5  # Warmup epochs
    
    # Milestone epochs where masking increases significantly - boost learning rate
    milestones = [int(num_epochs * x) for x in [0.2, 0.4, 0.6, 0.8]]
    boost_factor = 1.5
    
    # Check if at milestone
    at_milestone = epoch in milestones
    
    if epoch < warmup:
        # Warmup phase
        return base_lr * (epoch + 1) / warmup
    elif at_milestone:
        # Boost learning rate at masking milestones
        return min(base_lr * boost_factor, 1e-3)
    else:
        # Cosine decay with minimum lr
        progress = (epoch - warmup) / max(1, (num_epochs - warmup))
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine_decay

def visualize_reconstructions(model, dataloader, device, epoch, num_masks=None):
    """
    Create visualizations for wandb logging - both partial masking reconstructions
    and autoregressive generation.
    
    Args:
        model: Trained model
        dataloader: DataLoader with test data
        device: Device to use
        epoch: Current epoch
        num_masks: Number of masks to apply (optional)
    
    Returns:
        dict: Dictionary of images for wandb logging
    """
    from autoregressive_generation import generate_image_autoregressively
    
    model.eval()
    
    # Get a batch of data
    patches = next(iter(dataloader)).to(device)
    batch_size = min(4, patches.size(0))  # Limit to 4 images for visualization
    patches = patches[:batch_size]
    
    # Apply normal masking
    if num_masks is None:
        num_masks = min(64, calculate_mask_count(epoch, 1000, max_patches=256))
    
    # Adjust partial mask ratio for high masking rates
    if num_masks > 128:  # When masking more than 50% of patches
        partial_mask_ratio = max(0.1, min(0.5, 32 / num_masks))
    else:
        partial_mask_ratio = 0.3
    
    # Create masking info
    masking_info = model.apply_masking(
        patches, 
        num_masks=num_masks,
        partial_mask_ratio=partial_mask_ratio
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
        
        # For fully masked patches, set to gray (0.5)
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
    
    # Generate images autoregressively (1 sample)
    try:
        generated_image = generate_image_autoregressively(model, device, temperature=1.0)
        images["autoregressive_generation"] = wandb.Image(
            generated_image.cpu(), 
            caption="Autoregressive Generation"
        )
    except Exception as e:
        print(f"Error generating autoregressive image: {e}")
    
    # Also generate images in parallel (non-autoregressive) if we're at high masking rates
    if num_masks >= 128:
        try:
            patches_zero = torch.zeros((1, 256, 16), device=device)
            mask_full = torch.ones((1, 256), dtype=torch.bool, device=device)
            
            with torch.no_grad():
                outputs = model(patches_zero, mask_indices=mask_full)
                parallel_gen = model.reconstruct_image(outputs['binary_prediction'][0]).cpu()
                
            images["parallel_generation"] = wandb.Image(
                parallel_gen,
                caption="Parallel (Non-autoregressive) Generation"
            )
        except Exception as e:
            print(f"Error generating parallel image: {e}")
    
    return images
