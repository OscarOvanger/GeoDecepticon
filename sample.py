import torch
from transformer import VisionTransformer
from dataloader import preprocess_image

# Load model
#model = VisionTransformer(embed_dim=4, num_heads=2, feedforward_dim=8, num_layers=2, num_tokens=16, max_patches=1024)
#model.load_state_dict(torch.load("vision_transformer.pth"))
#model.eval()

def sample(image, model, device):
    patch_indices = preprocess_image(image).to(device)
    # Mask or partially mask patches here if needed
    logits = model(patch_indices, mask=torch.zeros_like(patch_indices, dtype=torch.bool))
    predictions = torch.argmax(logits, dim=-1)
    return predictions

def sequential_inpainting(model, input_image, mask_ratio=0.9, device="cpu"):
    """
    Sequentially inpaints an image by filling masked patches.
    
    Args:
        model (VisionTransformer): The trained Vision Transformer.
        input_image (Tensor): Input binary image of shape (64, 64).
        mask_ratio (float): Fraction of patches to mask initially.
        device (str): Device to run the inference on ("cpu" or "cuda").
    
    Returns:
        inpainted_image (Tensor): The inpainted binary image.
    """
    model.eval()
    model.to(device)

    # Preprocess image into patch indices
    patches = preprocess_image(input_image).to(device)

    # Mask patches
    num_patches = patches.size(0)
    mask = torch.rand(num_patches) < mask_ratio  # Random mask
    masked_patches = patches.clone()
    masked_patches[mask] = model.embedding_matrix.num_embeddings - 1  # Mask token index

    # Sequentially inpaint the masked patches
    while mask.any():
        with torch.no_grad():
            logits = model(masked_patches.unsqueeze(0).long(), mask.unsqueeze(0))  # Add batch dim
            probabilities = model.get_probabilities(logits)  # (1, num_patches, num_tokens)

        # Compute max probabilities for masked patches
        masked_probs = probabilities[0][mask]
        max_probs, _ = masked_probs.max(dim=-1)

        # Normalize max probabilities for sampling
        normalized_probs = max_probs / max_probs.sum()
        selected_patch_idx = torch.multinomial(normalized_probs, 1).item()

        # Sample a token for the selected patch
        token_distribution = masked_probs[selected_patch_idx]
        sampled_token = torch.multinomial(token_distribution, 1).item()

        # Update the input and mask
        patch_position = torch.where(mask)[0][selected_patch_idx]
        masked_patches[patch_position] = sampled_token
        mask[patch_position] = 0  # Unmask this patch

    # Convert back to image format (optional for visualization)
    reconstructed_image = reconstruct_image_from_patches(masked_patches)
    return reconstructed_image

def reconstruct_image_from_patches(patches):
    """
    Reconstructs a binary 64x64 image from 2x2 patches.

    Args:
        patches (Tensor): Patch indices of shape (num_patches,).
    
    Returns:
        image (Tensor): Reconstructed binary image of shape (64, 64).
    """
    patches = patches.to(torch.int)

    # Handle negative values (-1 for masked patches)
    mask = patches == -1
    patches[mask] = 0  # Temporarily set masked patches to 0 for processing

    # Convert indices back to binary patches
    patch_binary = ((patches.unsqueeze(1) & torch.tensor([8, 4, 2, 1], dtype=torch.int)) > 0).float()

    # Reshape binary patches into the original image structure
    image = patch_binary.view(32, 32, 2, 2).permute(0, 2, 1, 3).reshape(64, 64)

    # Restore masked patches as distinct value (-1)
    image[mask.view(32, 32, 2, 2).permute(0, 2, 1, 3).reshape(64, 64)] = -1
    return image


#from sample import sequential_inpainting
from transformer import VisionTransformer
from dataloader import preprocess_image
import torch
import matplotlib.pyplot as plt

def test_sequential_inpainting():
    print("Testing sequential inpainting...")

    # Create a dummy binary image (64x64)
    test_image = (torch.rand(64, 64) > 0.5).float()

    # Initialize a dummy model
    embed_dim = 4
    num_tokens = 16
    model = VisionTransformer(embed_dim, num_heads=2, feedforward_dim=8, num_layers=2, num_tokens=num_tokens, max_patches=32 * 32)

    # Perform sequential inpainting
    mask_ratio = 0.9
    inpainted_image = sequential_inpainting(model, test_image, mask_ratio=mask_ratio, device="cpu")

    # Check the shape of the inpainted image
    assert inpainted_image.shape == (64, 64), "Inpainted image shape is incorrect"

    # Ensure the image contains only binary values
    assert inpainted_image.min() >= 0 and inpainted_image.max() <= 1, "Inpainted image should contain binary values"

    # Visualize the original and inpainted images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Masked Image")
    plt.imshow(test_image, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Inpainted Image")
    plt.imshow(inpainted_image, cmap="gray")
    plt.axis("off")

    plt.show()

    print("Sequential inpainting test passed!")

#test_sequential_inpainting()
