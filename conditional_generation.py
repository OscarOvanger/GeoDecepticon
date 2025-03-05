import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def generate_from_pixel_conditions(model, condition_indices, condition_values, device='cuda', num_samples=3):
    """
    Generate images from pixel conditions using autoregressive generation and calculate log probabilities.
    
    Args:
        model: Trained model
        condition_indices: 1D indices in flattened 64x64 image
        condition_values: Binary values for those indices
        device: Device to use
        num_samples: Number of samples to generate
        
    Returns:
        list: List of (image, log_prob) tuples
    """
    model.eval()
    
    # Convert to numpy arrays for easier indexing
    condition_indices = np.array(condition_indices)
    condition_values = np.array(condition_values)
    
    image_size = 64
    patch_size = 4
    
    # Results list
    results = []
    
    for _ in range(num_samples):
        # Initialize empty image
        image = torch.zeros(image_size, image_size, device=device)
        
        # Set the conditional pixels
        for idx, val in zip(condition_indices, condition_values):
            i = idx // image_size
            j = idx % image_size
            image[i, j] = val
        
        # Create a mask of pixels to generate (True = generate, False = conditional)
        pixel_mask = torch.ones(image_size, image_size, dtype=torch.bool, device=device)
        for idx in condition_indices:
            i = idx // image_size
            j = idx % image_size
            pixel_mask[i, j] = False
        
        # Get indices of pixels to generate (in random order for better mixing)
        pixels_to_generate = torch.nonzero(pixel_mask, as_tuple=True)
        pixel_indices = list(zip(pixels_to_generate[0].tolist(), pixels_to_generate[1].tolist()))
        np.random.shuffle(pixel_indices)
        
        # Log probability accumulator
        log_prob = 0.0
        
        # Generate pixels one by one
        for i, j in pixel_indices:
            # Convert image to patches
            patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
            patches = patches.contiguous().view(-1, patch_size*patch_size)
            patches = patches.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            with torch.no_grad():
                outputs = model(patches)
            
            # Calculate which patch and position this pixel belongs to
            patch_i = i // patch_size
            patch_j = j // patch_size
            patch_idx = patch_i * (image_size // patch_size) + patch_j
            
            local_i = i % patch_size
            local_j = j % patch_size
            pixel_pos = local_i * patch_size + local_j
            
            # Get probability for this pixel
            prob = outputs['probabilities'][0, patch_idx, pixel_pos].item()
            
            # Determine value (using threshold)
            value = 1.0 if prob > 0.5 else 0.0
            
            # Update image
            image[i, j] = value
            
            # Update log probability
            if value > 0.5:  # Generated 1
                log_prob += np.log(prob)
            else:  # Generated 0
                log_prob += np.log(1 - prob)
        
        # Add to results
        results.append((image, log_prob))
    
    return results

def generate_conditional_samples(model, device, condition_indices, condition_values, num_samples=3):
    """
    Generate and visualize samples conditioned on specific pixel values.
    
    Args:
        model: Trained model
        device: Computation device
        condition_indices: 1D indices in flattened image
        condition_values: Binary values for those indices
        num_samples: Number of samples to generate
        
    Returns:
        dict: Dictionary of images for wandb logging
    """
    # Generate samples
    samples = generate_from_pixel_conditions(
        model, 
        condition_indices, 
        condition_values, 
        device, 
        num_samples
    )
    
    # Create visualization images for wandb
    images = {}
    
    # Create an image showing the conditional pixels
    image_size = 64
    conditional_img = np.zeros((image_size, image_size))
    for idx, val in zip(condition_indices, condition_values):
        i = idx // image_size
        j = idx % image_size
        conditional_img[i, j] = val
    
    images["conditional_pixels"] = wandb.Image(
        conditional_img,
        caption=f"Conditional Pixels ({len(condition_indices)} pixels)"
    )
    
    # Add each generated sample with its log probability
    for i, (img, log_prob) in enumerate(samples):
        images[f"conditional_sample_{i+1}"] = wandb.Image(
            img.cpu().numpy(),
            caption=f"Sample {i+1} - Log Prob: {log_prob:.2f}"
        )
        
    # Create a combined figure with all samples and log probabilities
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Show conditional pixels
    axes[0].imshow(conditional_img, cmap='gray')
    axes[0].set_title("Conditional Pixels")
    axes[0].axis('off')
    
    # Show each generated sample
    for i, (img, log_prob) in enumerate(samples):
        axes[i+1].imshow(img.cpu().numpy(), cmap='gray')
        axes[i+1].set_title(f"Sample {i+1}\nLog Prob: {log_prob:.2f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    
    # Add the figure to the images dictionary
    images["combined_samples"] = wandb.Image(plt)
    plt.close(fig)
    
    return images

def visualize_conditional_generation(samples, condition_indices, condition_values, image_size=64):
    """
    Create a standalone visualization of conditional generation.
    
    Args:
        samples: List of (image, log_prob) tuples
        condition_indices: 1D indices in flattened image
        condition_values: Binary values for those indices
        image_size: Size of the image
        
    Returns:
        Figure: Matplotlib figure with visualization
    """
    # Create an image showing the conditional pixels
    conditional_img = np.zeros((image_size, image_size))
    for idx, val in zip(condition_indices, condition_values):
        i = idx // image_size
        j = idx % image_size
        conditional_img[i, j] = val
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Show conditional pixels
    axes[0].imshow(conditional_img, cmap='gray')
    axes[0].set_title(f"Conditional Pixels ({len(condition_indices)} pixels)")
    axes[0].axis('off')
    
    # Show each generated sample
    for i, (img, log_prob) in enumerate(samples):
        if isinstance(img, torch.Tensor):
            img = img.cpu().numpy()
        axes[i+1].imshow(img, cmap='gray')
        axes[i+1].set_title(f"Sample {i+1}\nLog Prob: {log_prob:.2f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig
