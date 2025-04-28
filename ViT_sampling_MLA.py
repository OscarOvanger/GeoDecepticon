import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from ViT_MLA import *

def generate_image_mla(
    model, patch_size, image_size,
    condition_indices=None, condition_values=None,
    generation_order="manhattan"
):
    model.eval()
    grid_size = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # (A) Initialize KV cache once for an all-mask input
    model.init_kv_cache(batch_size=1, total_patches=total_patches)

    # (B) Prepare generated_patches tensor
    gen_patches = model.mask_token.to(device).view(1,1,patch_dim) \
                   .expand(1, total_patches, patch_dim).clone()

    # (C) Build sampling order (same as before) …
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

    logsum = 0.0

    for pidx in sampling_order:
        # 1) Run forward FROM CACHE (no re‐projection of entire KV)
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        logits_cur = logits[0, pidx, :].clone()

        # 2) Apply conditioning mask (if any)
        if pidx in conditions_by_patch:
            valid = torch.ones(model.vocab_size, device=device, dtype=torch.bool)
            for (li, v) in conditions_by_patch[pidx]:
                valid &= (model.vocab[:, li].to(device) == v)
            logits_cur[~valid] = -float("inf")

        # 3) Sample & update
        probs = F.softmax(logits_cur, dim=-1)
        tidx = torch.multinomial(probs, 1).item()
        logsum += math.log(probs[tidx].item() + 1e-10)
        # new patch embedding after mask_proj+pos:
        new_patch = model.vocab[tidx].view(1, patch_dim).to(device)
        # write into gen_patches
        gen_patches[0, pidx] = new_patch

        # 4) **Update** KV‐cache for the new patch across all layers
        #    need x_j patch embedding at each layer’s input:
        #    we can re‐embed with patch_proj+pos once, then propagate through blocks 
        #    up to each block’s start. For simplicity here we
        #    assume the same `x_tokens` used in init_kv_cache:
        x_patch = model.patch_proj(new_patch) + model.pos_emb[pidx].unsqueeze(0)
        for layer_i in range(len(model.encoder_layers)):
            x_patch = block(x_patch.unsqueeze(1),attn_bias=None).squeeze(1)
            model.update_kv_cache(layer_i, pidx, x_patch)

    # Reconstruct final image
    final = patches_to_image(gen_patches[0].cpu(), (image_size, image_size), patch_size)
    return final, logsum

def generate_images_batch_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None,
    generation_order="manhattan"
):
    model.eval()
    grid_size = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # A) init KV cache for whole batch
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # B) init gen_patches [B,P,D]
    gen_patches = model.mask_token.to(device).view(1,total_patches,patch_dim) \
                   .expand(batch_size, total_patches, patch_dim).clone()

    # C) build sampling_order & conditions_by_patch as before…
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
    loglikes = [0.0]*batch_size

    for pidx in sampling_order:
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        logits_cur = logits[:, pidx, :].clone()  # [B, V]

        if pidx in conditions_by_patch:
            valid = torch.ones(model.vocab_size, device=device, dtype=torch.bool)
            for (li, v) in conditions_by_patch[pidx]:
                valid &= (model.vocab[:, li].to(device) == v)
            logits_cur[:, ~valid] = -float('inf')

        probs = F.softmax(logits_cur, dim=-1)  # [B, V]
        samp = torch.multinomial(probs, 1).squeeze(-1)  # [B]
        for i in range(batch_size):
            loglikes[i] += math.log(probs[i, samp[i]].item() + 1e-10)
        # update patches & caches
        new_patches = model.vocab[samp].to(device)  # [B, patch_dim]
        gen_patches[:, pidx] = new_patches
        # update each layer’s KV cache for pidx
        x_patch = model.patch_proj(new_patches) + model.pos_emb[pidx].unsqueeze(0)
        for li in range(len(model.encoder_layers)):
            x_patch = block(x_patch.unsqueeze(1), attn_bias=None).squeeze(1) 
            model.update_kv_cache(li, pidx, x_patch)

    # reconstruct
    imgs = [patches_to_image(gen_patches[i].cpu(), (image_size, image_size), patch_size)
            for i in range(batch_size)]
    return torch.stack(imgs), loglikes

def generate_images_batch_method1_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None
):
    model.eval()
    grid_size = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # A) init KV cache
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)
    # B) init gen_patches
    gen_patches = model.mask_token.to(device).view(1,1,patch_dim) \
                   .expand(batch_size, total_patches, patch_dim).clone()

    # C) prepare observed list & to_fill count…
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
    
    loglikes = [0.0]*batch_size

    # 1) fill observed as before (identical to above, but using forward_from_cache & update_kv_cache)
    if observed:
        with torch.no_grad():
            logits_obs, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)  # [B, P, V]
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
                loglikes[i] += math.log(probs[i, samp[i]].item() + 1e-10)
            # write sampled patch into gen_patches
            new_patches = model.vocab[samp].to(device)  # [B, patch_dim]
            gen_patches[torch.arange(batch_size), p, :] = new_patches
            #Update KV Cache
            x_patch = model.patch_proj(new_patches) + model.pos_emb[p].unsqueeze(0)
            for li in range(len(model.encoder_layers)):
                x_patch = block(x_patch.unsqueeze(1), attn_bias=None).squeeze(1) 
                model.update_kv_cache(li, p, x_patch)
                
    # 2) dynamic steps:
    to_fill = total_patches - len(observed)
    for _ in range(to_fill):
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)  # [B,P,V]
        flat_logits = logits.view(batch_size*total_patches, -1)
        flat_probs  = F.softmax(flat_logits, dim=-1)
        flat_samps  = torch.multinomial(flat_probs, 1).squeeze(-1)
        flat_lp     = flat_probs[torch.arange(batch_size*total_patches, device=device), flat_samps]

        B = batch_size
        P = total_patches
        samps = flat_samps.view(B, P)   # [B, P]
        pliks = flat_lp.view(B, P)    # [B, P]
        
        # for filled positions, zero out so they won't be picked
        pliks[filled] = -float('inf')
        
        # for each image, pick the patch with highest sampled prob
        choice_patches = torch.argmax(pliks, dim=1)  # [B]
        
        # gather sampled_vocab_idx and accumulate log‐lik
        batch_idx = torch.arange(B, device=device)
        sel_vocab = samps[batch_idx, choice_patches]  # [B]
        for i in range(B):
            loglikes[i] += math.log(pliks[i, choice_patches[i]].item() + 1e-10)
        
        # write into gen_patches
        new_patches = model.vocab[sel_vocab].to(device)
        gen_patches[batch_idx, choice_patches, :] = model.vocab[sel_vocab].to(device)
        # mark filled
        filled[batch_idx, choice_patches] = True
        #Update KV Cache
        x_patch = model.patch_proj(new_patches) + model.pos_emb[choice_patches].unsqueeze(0)
        for li in range(len(model.encoder_layers)):
            x_patch = block(x_patch.unsqueeze(1), attn_bias=None).squeeze(1) 
            model.update_kv_cache(li, choice_patches, x_patch)
    
    # reconstruct images from patches
    gen_images = []
    for i in range(batch_size):
        img = patches_to_image(gen_patches[i].cpu(), (image_size, image_size), patch_size)
        gen_images.append(img)
    gen_images = torch.stack(gen_images)  # [B, H, W]
    
    return gen_images, log_liks
