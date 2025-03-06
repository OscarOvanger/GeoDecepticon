import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_conditional_images(model, condition_indices, condition_values, num_samples=3, device='cuda', temperature=1.0):
    """
    Generate images conditioned on specific pixel values using autoregressive generation.
    Tracks and returns log probabilities of generated images.
    
    Args:
        model: Trained ContinuousVisionTransformer model
        condition_indices: 1D indices in flattened 64x64 image
        condition_values: Binary values for those indices
        num_samples: Number of samples to generate
        device: Computation device
        temperature: Temperature for sampling (higher = more random)
        
    Returns:
        tuple: (list of generated images, list of log probabilities)
    """
    model.eval()
    
    # Image parameters
    image_size = 64
    patch_size = 4
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size
    
    # Results
    generated_images = []
    log_probs = []
    
    for sample_idx in range(num_samples):
        # Initialize empty image
        image = torch.zeros(image_size, image_size, device=device)
        
        # Set conditional pixels
        for idx, val in zip(condition_indices, condition_values):
            i, j = idx // image_size, idx % image_size
            image[i, j] = float(val)
        
        # Create mask for pixels to generate (True = need to generate)
        generation_mask = torch.ones(image_size, image_size, dtype=torch.bool, device=device)
        
        # Mark conditional pixels as already set
        for idx in condition_indices:
            i, j = idx // image_size, idx % image_size
            generation_mask[i, j] = False
        
        # Get all pixel positions that need to be generated
        pixels_to_generate = torch.nonzero(generation_mask, as_tuple=True)
        pixels_to_generate = list(zip(pixels_to_generate[0].tolist(), pixels_to_generate[1].tolist()))
        
        # Shuffle to improve mixing (optional)
        np.random.shuffle(pixels_to_generate)
        
        # Track log probability
        log_prob = 0.0
        
        # Generate pixels one by one
        for i, j in pixels_to_generate:
            # Extract patches from current image state
            patches = extract_patches(image, patch_size)
            patches = patches.unsqueeze(0)  # Add batch dimension
            
            # Forward pass
            with torch.no_grad():
                outputs = model(patches)
            
            # Find which patch and position this pixel belongs to
            patch_i, patch_j = i // patch_size, j // patch_size
            patch_idx = patch_i * (image_size // patch_size) + patch_j
            
            # Find position within patch
            local_i, local_j = i % patch_size, j % patch_size
            pixel_pos = local_i * patch_size + local_j
            
            # Get probability for this pixel
            prob = outputs['probabilities'][0, patch_idx, pixel_pos].item()
            
            # Apply temperature
            if temperature != 1.0:
                # Scale logit by temperature
                logit = torch.logit(torch.tensor(prob))
                prob = torch.sigmoid(logit / temperature).item()
            
            # Sample value
            if temperature > 0:
                # Probabilistic sampling
                value = float(torch.bernoulli(torch.tensor(prob)).item())
            else:
                # Deterministic (greedy) sampling
                value = 1.0 if prob > 0.5 else 0.0
            
            # Update image
            image[i, j] = value
            
            # Update log probability
            if value > 0.5:  # Generated a 1
                log_prob += np.log(prob)
            else:  # Generated a 0
                log_prob += np.log(1 - prob)
        
        # Add results
        generated_images.append(image.cpu())
        log_probs.append(log_prob)
    
    return generated_images, log_probs

def extract_patches(image, patch_size=4):
    """
    Extract 4x4 patches from image.
    
    Args:
        image: Tensor of shape [height, width]
        patch_size: Size of patches
        
    Returns:
        Tensor of shape [num_patches, patch_size*patch_size]
    """
    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(-1, patch_size*patch_size)
    return patches

def visualize_conditional_samples(images, log_probs, condition_indices, condition_values, image_size=64):
    """
    Visualize conditional samples with log probabilities.
    
    Args:
        images: List of generated images
        log_probs: List of log probabilities
        condition_indices: 1D indices of conditional pixels
        condition_values: Values for conditional pixels
        image_size: Size of square image
        
    Returns:
        matplotlib figure
    """
    num_samples = len(images)
    
    # Create condition visualization
    condition_img = np.zeros((image_size, image_size))
    for idx, val in zip(condition_indices, condition_values):
        i, j = idx // image_size, idx % image_size
        condition_img[i, j] = val
    
    # Create figure
    fig, axes = plt.subplots(1, num_samples + 1, figsize=(5 * (num_samples + 1), 5))
    
    # Plot conditional pixels
    axes[0].imshow(condition_img, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title(f"Conditional Pixels\n({len(condition_indices)} pixels)")
    axes[0].axis('off')
    
    # Plot each generated sample
    for i in range(num_samples):
        axes[i+1].imshow(images[i], cmap='gray', vmin=0, vmax=1)
        axes[i+1].set_title(f"Sample {i+1}\nLog Prob: {log_probs[i]:.2f}")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    return fig

def sample_from_model(model, load_path=None, device='cuda', num_samples=3):
    """
    Generate samples from a trained model.
    
    Args:
        model: ContinuousVisionTransformer model
        load_path: Path to model checkpoint (optional)
        device: Computation device
        num_samples: Number of samples to generate
        
    Returns:
        List of generated images
    """
    # Load model checkpoint if provided
    if load_path:
        model.load_state_dict(torch.load(load_path, map_location=device))
    
    model.eval()
    
    # Image parameters
    image_size = 64
    patch_size = 4
    num_patches = (image_size // patch_size) ** 2
    
    # Initialize empty images
    images = []
    
    for _ in range(num_samples):
        # Start with all zeros
        image = torch.zeros(image_size, image_size, device=device)
        
        # Generate image pixel by pixel
        for i in range(image_size):
            for j in range(image_size):
                # Extract patches from current image state
                patches = extract_patches(image, patch_size)
                patches = patches.unsqueeze(0)  # Add batch dimension
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(patches)
                
                # Find which patch and position this pixel belongs to
                patch_i, patch_j = i // patch_size, j // patch_size
                patch_idx = patch_i * (image_size // patch_size) + patch_j
                
                # Find position within patch
                local_i, local_j = i % patch_size, j % patch_size
                pixel_pos = local_i * patch_size + local_j
                
                # Get probability for this pixel
                prob = outputs['probabilities'][0, patch_idx, pixel_pos].item()
                
                # Sample value
                value = float(torch.bernoulli(torch.tensor(prob)).item())
                
                # Update image
                image[i, j] = value
        
        images.append(image.cpu())
    
    return images
