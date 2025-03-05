import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_image_autoregressively(model, device='cuda', temperature=1.0, size=64):
    """
    Generate an image one patch at a time, conditioning each new patch on previously generated ones.
    
    Args:
        model: Trained ContinuousVisionTransformer model
        device: Device to use
        temperature: Temperature for sampling (higher = more random)
        size: Image size (default 64x64)
        
    Returns:
        torch.Tensor: Generated image
    """
    model.eval()
    patch_size = 4
    num_patches = (size // patch_size) ** 2  # 256 for 64x64 image with 4x4 patches
    patch_dim = 16  # 4x4 patches flattened
    
    # Start with all masked patches
    patches = torch.zeros((1, num_patches, patch_dim), device=device)
    
    # Create mask tensor (all True initially - all patches masked)
    mask = torch.ones((1, num_patches), dtype=torch.bool, device=device)
    
    # Create partial mask tensors (empty initially)
    partial_mask = torch.zeros((1, num_patches), dtype=torch.bool, device=device)
    partial_values = torch.zeros((1, 1, patch_dim), device=device)
    
    # Define generation order - can be:
    # 1. Raster scan (left-to-right, top-to-bottom)
    # 2. Random order
    # 3. Spiral from center
    # 4. Based on patch entropy/uncertainty
    
    # Here we'll use a simple raster scan order
    generation_order = list(range(num_patches))
    
    # For visualization (optional)
    if temperature == 0:  # Only visualize for deterministic generation
        intermediate_images = []
    
    # Generate patches autoregressively
    for step, idx in enumerate(generation_order):
        # Current state: some patches generated, others still masked
        with torch.no_grad():
            outputs = model(
                patches, 
                mask_indices=mask,
                partial_mask_indices=partial_mask,
                partial_mask_values=partial_values
            )
        
        # Get prediction for current patch
        logits = outputs['logits'][0, idx]
        probs = outputs['probabilities'][0, idx]
        
        # Apply temperature for sampling
        if temperature > 0:
            # Scale logits by temperature
            scaled_logits = logits / temperature
            # Get probabilities
            scaled_probs = torch.sigmoid(scaled_logits)
            # Sample from the distribution
            patch_prediction = torch.bernoulli(scaled_probs).float()
        else:
            # Deterministic - use threshold at 0.5
            patch_prediction = (probs > 0.5).float()
        
        # Update the patches tensor with the new prediction
        patches[0, idx] = patch_prediction
        
        # Update mask to reflect that this patch is now generated
        mask[0, idx] = False
        
        # Visualize intermediate generation (optional)
        if temperature == 0 and (step % 16 == 0 or step == num_patches - 1):
            with torch.no_grad():
                current_image = model.reconstruct_image(patches[0]).cpu()
                # Create a visualization showing which patches are masked
                mask_vis = mask[0].cpu().view(int(size/patch_size), int(size/patch_size))
                # Expand to image size
                mask_image = torch.ones((size, size))
                for i in range(int(size/patch_size)):
                    for j in range(int(size/patch_size)):
                        if mask_vis[i, j]:
                            mask_image[i*patch_size:(i+1)*patch_size, 
                                       j*patch_size:(j+1)*patch_size] = 0.5
                
                intermediate_images.append((current_image, mask_image, step))
    
    # Reconstruct the final image
    with torch.no_grad():
        generated_image = model.reconstruct_image(patches[0])
    
    # Return visualizations if available
    if temperature == 0 and len(intermediate_images) > 0:
        return generated_image, intermediate_images
    else:
        return generated_image

def generate_conditional_image(model, initial_patches, initial_mask, device='cuda', temperature=1.0):
    """
    Generate an image conditioned on some initial patches.
    
    Args:
        model: Trained ContinuousVisionTransformer model
        initial_patches: Tensor with known patches [1, num_patches, patch_dim]
        initial_mask: Boolean tensor indicating which patches are masked (True) [1, num_patches]
        device: Device to use
        temperature: Temperature for sampling (higher = more random)
        
    Returns:
        torch.Tensor: Generated image
    """
    model.eval()
    
    # Start with given patches and mask
    patches = initial_patches.clone().to(device)
    mask = initial_mask.clone().to(device)
    
    # Get indices of masked patches
    masked_indices = torch.where(mask[0])[0].tolist()
    
    # Shuffle the order for more natural generation
    np.random.shuffle(masked_indices)
    
    # For visualization (optional)
    if temperature == 0:
        intermediate_images = []
    
    # Generate each masked patch
    for step, idx in enumerate(masked_indices):
        # Forward pass with current state
        with torch.no_grad():
            outputs = model(patches, mask_indices=mask)
        
        # Get prediction for current patch
        logits = outputs['logits'][0, idx]
        probs = outputs['probabilities'][0, idx]
        
        # Apply temperature for sampling
        if temperature > 0:
            # Scale logits by temperature
            scaled_logits = logits / temperature
            # Get probabilities
            scaled_probs = torch.sigmoid(scaled_logits)
            # Sample from the distribution
            patch_prediction = torch.bernoulli(scaled_probs).float()
        else:
            # Deterministic - use threshold at 0.5
            patch_prediction = (probs > 0.5).float()
        
        # Update current patch
        patches[0, idx] = patch_prediction
        
        # Update mask
        mask[0, idx] = False
        
        # Visualize intermediate generation (optional)
        if temperature == 0 and (step % 16 == 0 or step == len(masked_indices) - 1):
            with torch.no_grad():
                current_image = model.reconstruct_image(patches[0]).cpu()
                intermediate_images.append((current_image, step))
    
    # Reconstruct the final image
    with torch.no_grad():
        generated_image = model.reconstruct_image(patches[0])
    
    # Return visualizations if available
    if temperature == 0 and len(intermediate_images) > 0:
        return generated_image, intermediate_images
    else:
        return generated_image

def generate_images_batch(model, num_images=4, device='cuda', method='autoregressive', temperature=1.0):
    """
    Generate multiple images using autoregressive generation.
    
    Args:
        model: Trained ContinuousVisionTransformer model
        num_images: Number of images to generate
        device: Device to use
        method: 'autoregressive' or 'parallel' (non-autoregressive)
        temperature: Temperature for sampling (higher = more random)
        
    Returns:
        torch.Tensor: Batch of generated images [num_images, 64, 64]
    """
    generated_images = []
    
    if method == 'autoregressive':
        # Generate each image autoregressively
        for _ in range(num_images):
            image = generate_image_autoregressively(model, device, temperature)
            generated_images.append(image)
    else:
        # Non-autoregressive generation (original method)
        patch_size = 4
        size = 64
        num_patches = (size // patch_size) ** 2
        patch_dim = 16
        
        # Create dummy patches (all zeros)
        patches = torch.zeros((num_images, num_patches, patch_dim), device=device)
        
        # Create mask for all patches (100% masking)
        mask = torch.ones((num_images, num_patches), dtype=torch.bool, device=device)
        
        # No partial masking
        partial_mask = torch.zeros((num_images, num_patches), dtype=torch.bool, device=device)
        
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
        for i in range(num_images):
            image = model.reconstruct_image(binary_pred[i])
            generated_images.append(image)
    
    return torch.stack(generated_images)

def visualize_generation_process(intermediate_images, save_path=None):
    """
    Visualize the autoregressive image generation process.
    
    Args:
        intermediate_images: List of (image, mask_image, step) tuples
        save_path: Path to save the visualization (optional)
    """
    num_steps = len(intermediate_images)
    fig, axes = plt.subplots(num_steps, 2, figsize=(10, 2.5 * num_steps))
    
    for i, (image, mask_image, step) in enumerate(intermediate_images):
        # Plot the current state of the image
        axes[i, 0].imshow(image.numpy(), cmap='gray')
        axes[i, 0].set_title(f"Generation Step {step+1}")
        axes[i, 0].axis('off')
        
        # Plot the mask visualization
        axes[i, 1].imshow(mask_image.numpy(), cmap='gray')
        axes[i, 1].set_title(f"Remaining Masks: {mask_image.eq(0.5).sum().item()//16} patches")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def create_mask_pattern(pattern_type, height=64, width=64, patch_size=4):
    """
    Create different mask patterns for image inpainting.
    
    Args:
        pattern_type: Type of mask pattern ('center', 'random', 'half', 'checkerboard', etc.)
        height: Image height
        width: Image width
        patch_size: Size of each patch
        
    Returns:
        torch.Tensor: Boolean mask tensor (True for masked regions)
    """
    h_patches = height // patch_size
    w_patches = width // patch_size
    num_patches = h_patches * w_patches
    
    if pattern_type == 'center':
        # Mask the center region (e.g., 32x32 for a 64x64 image)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        center_size = h_patches // 2
        start_h = h_patches // 4
        start_w = w_patches // 4
        
        for i in range(start_h, start_h + center_size):
            for j in range(start_w, start_w + center_size):
                idx = i * w_patches + j
                mask[idx] = True
    
    elif pattern_type == 'random':
        # Randomly mask 30-50% of patches
        mask_percentage = np.random.uniform(0.3, 0.5)
        num_masked = int(num_patches * mask_percentage)
        mask = torch.zeros(num_patches, dtype=torch.bool)
        masked_indices = torch.randperm(num_patches)[:num_masked]
        mask[masked_indices] = True
    
    elif pattern_type == 'half':
        # Mask left or right half
        mask = torch.zeros(num_patches, dtype=torch.bool)
        half = w_patches // 2
        
        # Randomly choose left or right
        left_side = np.random.choice([True, False])
        
        for i in range(h_patches):
            for j in range(w_patches):
                if (left_side and j < half) or (not left_side and j >= half):
                    idx = i * w_patches + j
                    mask[idx] = True
    
    elif pattern_type == 'checkerboard':
        # Create a checkerboard pattern
        mask = torch.zeros(num_patches, dtype=torch.bool)
        
        for i in range(h_patches):
            for j in range(w_patches):
                if (i + j) % 2 == 0:  # Alternate patches
                    idx = i * w_patches + j
                    mask[idx] = True
    
    elif pattern_type == 'vertical_lines':
        # Mask vertical lines
        mask = torch.zeros(num_patches, dtype=torch.bool)
        stride = 3  # Every 3rd column
        
        for i in range(h_patches):
            for j in range(0, w_patches, stride):
                idx = i * w_patches + j
                mask[idx] = True
    
    elif pattern_type == 'horizontal_lines':
        # Mask horizontal lines
        mask = torch.zeros(num_patches, dtype=torch.bool)
        stride = 3  # Every 3rd row
        
        for i in range(0, h_patches, stride):
            for j in range(w_patches):
                idx = i * w_patches + j
                mask[idx] = True
    
    else:
        raise ValueError(f"Unknown mask pattern: {pattern_type}")
    
    return mask

def inpaint_image(model, image, mask_pattern='center', device='cuda', temperature=0.0, autoregressive=True):
    """
    Perform image inpainting on the given image with specified mask pattern.
    
    Args:
        model: Trained ContinuousVisionTransformer model
        image: Image tensor [64, 64]
        mask_pattern: Type of mask to apply ('center', 'random', 'half', etc.)
        device: Device to use
        temperature: Temperature for sampling (0 = deterministic)
        autoregressive: Whether to use autoregressive generation
        
    Returns:
        tuple: (original_image, masked_image, inpainted_image)
    """
    model.eval()
    patch_size = 4
    height, width = image.shape
    h_patches = height // patch_size
    w_patches = width // patch_size
    num_patches = h_patches * w_patches
    patch_dim = patch_size * patch_size
    
    # Extract patches from the image
    patches = image.unfold(0, patch_size, patch_size).unfold(1, patch_size, patch_size)
    patches = patches.contiguous().view(num_patches, patch_dim)
    patches = patches.unsqueeze(0)  # Add batch dimension [1, num_patches, patch_dim]
    
    # Create mask
    mask = create_mask_pattern(mask_pattern, height, width, patch_size)
    mask = mask.unsqueeze(0)  # Add batch dimension [1, num_patches]
    
    # Original patches for reference
    original_patches = patches.clone()
    
    if autoregressive:
        # Use autoregressive inpainting
        result = generate_conditional_image(
            model, 
            original_patches, 
            mask,
            device=device, 
            temperature=temperature
        )
        
        # Check if we got visualizations
        if isinstance(result, tuple):
            inpainted_image, intermediate_steps = result
            return original_patches, mask, inpainted_image, intermediate_steps
        else:
            inpainted_image = result
            return original_patches, mask, inpainted_image
    else:
        # Non-autoregressive inpainting
        # Create masked patches
        masked_patches = original_patches.clone().to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                masked_patches,
                mask_indices=mask.to(device)
            )
        
        # Get predictions
        binary_pred = outputs['binary_prediction'][0]
        
        # Combine original patches with predictions for masked regions
        inpainted_patches = original_patches.clone()[0]
        inpainted_patches[mask[0]] = binary_pred[mask[0]]
        
        # Reconstruct image
        inpainted_image = model.reconstruct_image(inpainted_patches)
        
        # Create visualization of masked image
        masked_image = original_patches.clone()[0]
        masked_image[mask[0]] = torch.ones_like(masked_image[mask[0]]) * 0.5
        masked_image_viz = model.reconstruct_image(masked_image)
        
        return model.reconstruct_image(original_patches[0]), masked_image_viz, inpainted_image

def visualize_inpainting(original, masked, inpainted, title="Image Inpainting"):
    """
    Visualize the inpainting results.
    
    Args:
        original: Original image
        masked: Masked image
        inpainted: Inpainted image
        title: Plot title
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original.cpu().numpy(), cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(masked.cpu().numpy(), cmap='gray')
    axes[1].set_title("Masked Image")
    axes[1].axis('off')
    
    axes[2].imshow(inpainted.cpu().numpy(), cmap='gray')
    axes[2].set_title("Inpainted Image")
    axes[2].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    
    return fig

def compare_inpainting_methods(model, image, mask_pattern='random', device='cuda'):
    """
    Compare autoregressive vs non-autoregressive inpainting.
    
    Args:
        model: Trained model
        image: Input image
        mask_pattern: Type of mask to apply
        device: Computation device
    """
    # Non-autoregressive inpainting
    original_non_ar, masked_non_ar, inpainted_non_ar = inpaint_image(
        model, image, mask_pattern, device, temperature=0.0, autoregressive=False
    )
    
    # Autoregressive inpainting
    original_ar, masked_ar, inpainted_ar = inpaint_image(
        model, image, mask_pattern, device, temperature=0.0, autoregressive=True
    )
    
    # Visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Non-autoregressive results
    axes[0, 0].imshow(original_non_ar.cpu().numpy(), cmap='gray')
    axes[0, 0].set_title("Original Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(masked_non_ar.cpu().numpy(), cmap='gray')
    axes[0, 1].set_title("Masked Image")
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(inpainted_non_ar.cpu().numpy(), cmap='gray')
    axes[0, 2].set_title("Non-Autoregressive Inpainting")
    axes[0, 2].axis('off')
    
    # Autoregressive results
    axes[1, 0].imshow(original_ar.cpu().numpy(), cmap='gray')
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(masked_ar.cpu().numpy(), cmap='gray')
    axes[1, 1].set_title("Masked Image")
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(inpainted_ar.cpu().numpy(), cmap='gray')
    axes[1, 2].set_title("Autoregressive Inpainting")
    axes[1, 2].axis('off')
    
    plt.suptitle(f"Inpainting Comparison: {mask_pattern} mask pattern")
    plt.tight_layout()
    plt.show()
