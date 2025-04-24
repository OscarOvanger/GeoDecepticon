import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from ViT import *
########################################
# Generates conditional images
########################################

def generate_image(
    model,
    patch_size,
    image_size,
    condition_indices=None,
    condition_values=None,
    generation_order="manhattan"
):
    """
    Generate an image (as a tensor of shape [H, W]) conditionally based on observed cells,
    using different generation orders for unobserved patches.

    Args:
        model: The trained ViT model.
               Assumes model.vocab is a tensor of shape [vocab_size, patch_dim] and
               model.mask_token is a tensor of shape [patch_dim].
        patch_size: The side length of each square patch.
        image_size: The side length of the square image.
        condition_indices: A list (or tensor) of flattened full image indices where cells are observed.
        condition_values: A list (or tensor) of the same length as condition_indices containing
                          the observed cell values (e.g. 0 or 1).
        generation_order: One of {"manhattan", "raster", "random"} controlling how unobserved
                          patches are generated after the observed patches.

    Returns:
        generated_image: A tensor of shape [image_size, image_size] representing the generated image.
        log_likelihood_sum: The sum of the log-likelihoods for each patch sampled.
    """
    # Set model to evaluation mode.
    model.eval()

    # Compute grid dimensions.
    grid_size = image_size // patch_size  # assuming image_size is divisible by patch_size
    total_patches = grid_size * grid_size
    patch_dim = patch_size * patch_size

    # Create an initial "generated" representation: all patches are initially the mask token.
    # Shape: [total_patches, patch_dim]
    generated_patches = model.mask_token.expand(total_patches, patch_dim).clone()

    # Build a mapping from patch index -> list of (local_index, observed value)
    conditions_by_patch = {}
    if condition_indices is not None and condition_values is not None:
        for cond_idx, cond_val in zip(condition_indices, condition_values):
            # Convert the full-image index to (row, col)
            pixel_row = cond_idx // image_size
            pixel_col = cond_idx % image_size
            # Determine which patch this pixel falls into.
            patch_row = pixel_row // patch_size
            patch_col = pixel_col // patch_size
            patch_idx = patch_row * grid_size + patch_col
            # Determine cell (local) coordinates within the patch.
            local_row = pixel_row % patch_size
            local_col = pixel_col % patch_size
            local_idx = local_row * patch_size + local_col
            # Save this condition in the dictionary.
            if patch_idx not in conditions_by_patch:
                conditions_by_patch[patch_idx] = []
            conditions_by_patch[patch_idx].append((local_idx, cond_val))

    # 1) Observed patches first.
    observed_patch_ids = sorted(list(conditions_by_patch.keys()))

    # 2) Unobserved patches, order determined by generation_order.
    unobserved_patch_ids = []
    observed_rows = [(pid // grid_size) for pid in observed_patch_ids]
    observed_cols = [(pid % grid_size) for pid in observed_patch_ids]

    # Gather all unobserved patches:
    all_patch_indices = set(range(total_patches))
    for pid in observed_patch_ids:
        if pid in all_patch_indices:
            all_patch_indices.remove(pid)

    if generation_order == "manhattan":
        # For patches without conditions, compute Manhattan distance sum to all observed patches.
        distance_list = []
        for p in all_patch_indices:
            row = p // grid_size
            col = p % grid_size
            dist_sum = 0
            for op in observed_patch_ids:
                orow = op // grid_size
                ocol = op % grid_size
                dist_sum += abs(row - orow) + abs(col - ocol)
            distance_list.append((p, dist_sum))
        distance_list.sort(key=lambda x: x[1])
        unobserved_patch_ids = [p for p, _ in distance_list]

    elif generation_order == "raster":
        # Sort by patch index in ascending order (row-major).
        unobserved_patch_ids = sorted(list(all_patch_indices))

    elif generation_order == "random":
        import random
        unobserved_patch_ids = list(all_patch_indices)
        random.shuffle(unobserved_patch_ids)

    else:
        raise ValueError(f"Unknown generation_order: {generation_order}. "
                         "Must be one of ['manhattan', 'raster', 'random'].")

    # Combine into a final sampling order: observed first, then the unobserved in the chosen order.
    sampling_order = observed_patch_ids + unobserved_patch_ids

    log_likelihood_sum = 0.0

    # Now, generate patch-by-patch.
    # We use a batch size of 1 for generation.
    for p_idx in sampling_order:
        current_tokens = generated_patches.unsqueeze(0).clone()  # [1, total_patches, patch_dim]
        # During generation we do not want additional random masking, so pass mask_rate=0.
        with torch.no_grad():
            logits, _ = model(current_tokens, mask_rate=0.0)  # logits shape: [1, total_patches, vocab_size]
        logits_current = logits[0, p_idx]  # shape [vocab_size]

        # If this patch has observations, filter logits to keep only vocabulary entries that match.
        if p_idx in conditions_by_patch:
            valid_candidates = torch.ones(model.vocab.shape[0], dtype=torch.bool, device=logits.device)
            for local_idx, cond_val in conditions_by_patch[p_idx]:
                valid_candidates &= (model.vocab[:, local_idx] == cond_val)
            logits_current[~valid_candidates] = -float("inf")

        # Compute probabilities from the logits.
        probs = F.softmax(logits_current, dim=-1)
        # Sample an index from the distribution.
        sampled_idx = torch.multinomial(probs, num_samples=1).item()
        # Accumulate the log probability.
        log_prob = torch.log(probs[sampled_idx] + 1e-10).item()  # add epsilon for stability
        log_likelihood_sum += log_prob

        # Retrieve the sampled patch from the vocabulary.
        sampled_patch = model.vocab[sampled_idx]
        # Update the generated patches.
        generated_patches[p_idx] = sampled_patch

    # Reconstruct the full image from patches.
    generated_image = patches_to_image(generated_patches, (image_size, image_size), patch_size)

    return generated_image, log_likelihood_sum



def generate_images_batch(model, patch_size, image_size, batch_size,device, 
                          condition_indices=None, condition_values=None, 
                          generation_order="manhattan"):
    """
    Generate a batch of images (batch_size images) in parallel.
    If condition_indices and condition_values are provided, they apply to all images in the batch.
    (For simplicity, this implementation assumes the same conditioned pixel positions for all images.)
    Returns: (tensor of shape [batch_size, image_size, image_size], list of log-likelihoods)
    """
    model.eval()
    grid_size = image_size // patch_size
    total_patches = grid_size * grid_size
    patch_dim = patch_size * patch_size

    # Initialize all images' patches as mask token
    mask_patch = model.mask_token.detach()  # [patch_dim]
    # Create [batch_size, total_patches, patch_dim] filled with mask token
    generated_patches = mask_patch.unsqueeze(0).expand(total_patches, patch_dim)
    generated_patches = generated_patches.unsqueeze(0).expand(batch_size, total_patches, patch_dim).clone().to(device)

    # Prepare condition constraints for patches (same for each image for now)
    conditions_by_patch = {}
    observed_patch_ids = []
    if condition_indices is not None and condition_values is not None:
        condition_indices = np.array(condition_indices)
        condition_values = np.array(condition_values)
        if condition_values.ndim == 1:
            # Use the same condition values for all images
            condition_values_batch = np.tile(condition_values, (batch_size, 1))
        else:
            # If a different set of values per image is provided (condition_values shape [batch, k]),
            # we'll take the first one as representative for ordering (assuming indices same).
            condition_values_batch = condition_values
        # Build a dictionary of constraints for each patch (assuming all images share condition indices)
        for cond_idx, cond_val in zip(condition_indices, condition_values_batch[0]):
            pixel_row = cond_idx // image_size
            pixel_col = cond_idx % image_size
            patch_row = pixel_row // patch_size
            patch_col = pixel_col // patch_size
            patch_idx = patch_row * grid_size + patch_col
            local_row = pixel_row % patch_size
            local_col = pixel_col % patch_size
            local_idx = local_row * patch_size + local_col
            conditions_by_patch.setdefault(patch_idx, []).append((local_idx, cond_val))
        observed_patch_ids = sorted(conditions_by_patch.keys())
    else:
        condition_values_batch = None

    # Determine generation order for patches
    all_patches = set(range(total_patches))
    unobserved_patch_ids = list(all_patches - set(observed_patch_ids))
    if generation_order == "manhattan" and observed_patch_ids:
        distance_list = []
        for p in unobserved_patch_ids:
            row, col = divmod(p, grid_size)
            dist_sum = 0
            for op in observed_patch_ids:
                orow, ocol = divmod(op, grid_size)
                dist_sum += abs(row - orow) + abs(col - ocol)
            distance_list.append((p, dist_sum))
        distance_list.sort(key=lambda x: x[1])
        unobserved_patch_ids = [p for p, _ in distance_list]
    elif generation_order == "raster":
        unobserved_patch_ids.sort()
    elif generation_order == "random":
        np.random.shuffle(unobserved_patch_ids)
    else:
        raise ValueError(f"Unknown generation_order: {generation_order}")

    sampling_order = observed_patch_ids + unobserved_patch_ids

    # Initialize log-likelihood sums for each image in batch
    log_likelihoods = [0.0] * batch_size

    # Iteratively generate each patch for all images in parallel
    for p_idx in sampling_order:
        with torch.no_grad():
            # Forward pass for all images with current known patches
            logits, _ = model(generated_patches, mask_rate=0.0)  # [batch, total_patches, vocab_size]
        # logits for the current patch index for all images: shape [batch_size, vocab_size]
        logits_current = logits[:, p_idx, :].clone()

        # If this patch is conditioned (observed) and we have constraints:
        if p_idx in conditions_by_patch:
            # We restrict vocabulary choices for all images based on the condition.
            # (Assuming same condition constraint for all images.)
            valid_vocab = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
            for local_idx, val in conditions_by_patch[p_idx]:
                valid_vocab &= (model.vocab[:, local_idx] == val)
            # Set invalid vocab logits to -inf for all images
            logits_current[:, ~valid_vocab] = -float('inf')

        # Compute probabilities and sample a patch for each image in the batch
        probs = F.softmax(logits_current, dim=-1)  # [batch_size, vocab_size]
        sampled_indices = []
        for i in range(batch_size):
            idx = torch.multinomial(probs[i], num_samples=1).item()
            sampled_indices.append(idx)
            # Accumulate log-likelihood for each image
            log_likelihoods[i] += math.log(probs[i, idx].item() + 1e-10)
        sampled_indices = torch.tensor(sampled_indices, device=device)
        # Retrieve the actual patch values for these sampled vocab indices
        sampled_patches = model.vocab[sampled_indices]  # [batch_size, patch_dim]
        # Update the generated patches for all images at position p_idx
        generated_patches[:, p_idx, :] = sampled_patches

    # All patches filled, reconstruct images
    generated_images = []
    for i in range(batch_size):
        img = patches_to_image(generated_patches[i].cpu(), (image_size, image_size), patch_size)
        generated_images.append(img)
    generated_images = torch.stack(generated_images)
    return generated_images, log_likelihoods


def generate_images_batch_method1(
    model,
    patch_size: int,
    image_size: int,
    batch_size: int,
    device,
    condition_indices=None,
    condition_values=None
):
    """
    Method 1: 'sample‐all‐then‐select' batch generation.
    
    Args:
      model: a trained StackedContextViT on device.
      patch_size: size of one patch (e.g. 8).
      image_size: full image side length (e.g. 64).
      batch_size: how many images to generate in parallel.
      condition_indices: list of flat pixel indices to condition on (optional).
      condition_values: list of 0/1 values for each index (optional).
    
    Returns:
      generated_images: Tensor [batch_size, image_size, image_size]
      log_likelihoods: list of length batch_size
    """
    model.eval()
    device = next(model.parameters()).device
    
    # grid & patch dims
    grid_size    = image_size // patch_size
    total_patches = grid_size * grid_size
    patch_dim     = patch_size * patch_size
    
    # initialize all patches to the mask token
    mask_tok = model.mask_token.detach().to(device)             # [patch_dim]
    # [batch, total_patches, patch_dim]
    gen_patches = (
        mask_tok
        .unsqueeze(0)       # [1, patch_dim]
        .unsqueeze(0)       # [1, 1, patch_dim]
        .expand(batch_size, total_patches, patch_dim)
        .clone()            # [batch_size, total_patches, patch_dim]
    )
    
    # build conditioning dict: {patch_idx: [(local_idx, val), ...], ...}
    cond_by_patch = {}
    if condition_indices is not None and condition_values is not None:
        for pix_idx, val in zip(condition_indices, condition_values):
            row = pix_idx // image_size
            col = pix_idx %  image_size
            prow, pcol = row // patch_size, col // patch_size
            patch_idx = prow * grid_size + pcol
            local_r = row % patch_size
            local_c = col %  patch_size
            local_idx = local_r * patch_size + local_c
            cond_by_patch.setdefault(patch_idx, []).append((local_idx, val))
    observed = sorted(cond_by_patch.keys())
    
    # track which patches are already filled
    filled = torch.zeros(batch_size, total_patches, dtype=torch.bool, device=device)
    filled[:, observed] = True
    
    # initialize log‐likelihood accumulators
    log_liks = [0.0] * batch_size
    
    # 1) **First**, fill observed patches (sampling under constraints)
    if observed:
        with torch.no_grad():
            logits_obs, _ = model(gen_patches, mask_rate=0.0)  # [B, P, V]
        for p in observed:
            # logits for this patch: [B, V]
            lc = logits_obs[:, p, :].clone()
            # apply conditioning mask
            valid = torch.ones(model.vocab_size, dtype=torch.bool, device=device)
            for (lidx, v) in cond_by_patch[p]:
                valid &= (model.vocab[:, lidx] == v)
            lc[:, ~valid] = -float('inf')
            
            probs = F.softmax(lc, dim=-1)              # [B, V]
            samp = torch.multinomial(probs, num_samples=1).squeeze(-1)  # [B]
            # update log‐likelihoods
            for i in range(batch_size):
                log_liks[i] += math.log(probs[i, samp[i]].item() + 1e-10)
            # write sampled patch into gen_patches
            gen_patches[torch.arange(batch_size), p, :] = model.vocab[samp].to(device)
    
    # 2) **Then**, iteratively fill the remaining patches
    to_fill = total_patches - len(observed)
    for _ in range(to_fill):
        with torch.no_grad():
            logits, _ = model(gen_patches, mask_rate=0.0)  # [B, P, V]
        
        B, P, V = logits.shape
        # flatten for multinomial
        flat_logits = logits.view(B*P, V)
        flat_probs  = F.softmax(flat_logits, dim=-1)
        flat_samps  = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # [B*P]
        flat_liks   = flat_probs[torch.arange(B*P, device=device), flat_samps]
        
        # reshape back
        samps = flat_samps.view(B, P)   # [B, P]
        pliks = flat_liks.view(B, P)    # [B, P]
        
        # for filled positions, zero out so they won't be picked
        pliks[filled] = -float('inf')
        
        # for each image, pick the patch with highest sampled prob
        choice_patches = torch.argmax(pliks, dim=1)  # [B]
        
        # gather sampled_vocab_idx and accumulate log‐lik
        batch_idx = torch.arange(B, device=device)
        sel_vocab = samps[batch_idx, choice_patches]  # [B]
        for i in range(B):
            log_liks[i] += math.log(pliks[i, choice_patches[i]].item() + 1e-10)
        
        # write into gen_patches
        gen_patches[batch_idx, choice_patches, :] = model.vocab[sel_vocab].to(device)
        # mark filled
        filled[batch_idx, choice_patches] = True
    
    # reconstruct images from patches
    gen_images = []
    for i in range(batch_size):
        img = patches_to_image(gen_patches[i].cpu(), (image_size, image_size), patch_size)
        gen_images.append(img)
    gen_images = torch.stack(gen_images)  # [B, H, W]
    
    return gen_images, log_liks
