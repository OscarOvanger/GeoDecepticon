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

