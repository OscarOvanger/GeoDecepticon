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
    device = next(model.parameters()).device

    grid_size     = image_size // patch_size
    total_patches = grid_size * grid_size
    patch_dim     = patch_size * patch_size

    # A) first, cache the all-masked state
    model.init_kv_cache(batch_size=1, total_patches=total_patches)

    # B) gen_patches: [1, total_patches, patch_dim]
    gen_patches = (
        model.mask_token.to(device)
        .view(1, total_patches, patch_dim)
        .expand(1, total_patches, patch_dim)
        .clone()
    )

    # C) build conditions_by_patch & sampling_order (same logic you already have)
    conditions_by_patch = {}
    if condition_indices is not None:
        for idx, val in zip(condition_indices, condition_values):
            r, c = divmod(idx, image_size)
            pr, pc = r // patch_size, c // patch_size
            pidx = pr * grid_size + pc
            local = (r % patch_size) * patch_size + (c % patch_size)
            conditions_by_patch.setdefault(pidx, []).append((local, val))

    observed = sorted(conditions_by_patch.keys())
    all_patches = set(range(total_patches))
    unobs = list(all_patches - set(observed))
    if generation_order == "manhattan" and observed:
        dist_list = []
        for p in unobs:
            pr, pc = divmod(p, grid_size)
            s = sum(abs(pr - (op // grid_size)) + abs(pc - (op % grid_size)) for op in observed)
            dist_list.append((p, s))
        dist_list.sort(key=lambda x: x[1])
        unobs = [p for p,_ in dist_list]
    elif generation_order == "raster":
        unobs.sort()
    elif generation_order == "random":
        np.random.shuffle(unobs)
    else:
        raise ValueError(generation_order)
    sampling_order = observed + unobs

    logsum = 0.0
    for pidx in sampling_order:
        # 1) fast forward using cached KV:
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        cur_logits = logits[0, pidx].clone()

        # 2) apply any condition mask
        if pidx in conditions_by_patch:
            valid = torch.ones(model.vocab_size, device=device, dtype=torch.bool)
            for local, val in conditions_by_patch[pidx]:
                valid &= (model.vocab[:, local].to(device) == val)
            cur_logits[~valid] = -float('inf')

        # 3) sample
        probs = F.softmax(cur_logits, dim=-1)
        tidx  = torch.multinomial(probs, 1).item()
        logsum += math.log(probs[tidx].item() + 1e-10)

        # 4) write patch back into gen_patches
        gen_patches[0, pidx] = model.vocab[tidx].to(device)

        # 5) **re-init** the cache so KV exactly matches the new gen_patches
        model.init_kv_cache(batch_size=1, total_patches=total_patches)

    # reconstruct and return
    final = patches_to_image(gen_patches[0].cpu(), (image_size, image_size), patch_size)
    return final, logsum


def generate_images_batch_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None,
    generation_order="manhattan"
):
    model.eval()
    device = next(model.parameters()).device

    grid_size     = image_size // patch_size
    total_patches = grid_size * grid_size
    patch_dim     = patch_size * patch_size

    # A) init cache for the batch
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # B) init gen_patches [B, P, D]
    gen_patches = (
        model.mask_token.to(device)
        .view(1, total_patches, patch_dim)
        .expand(batch_size, total_patches, patch_dim)
        .clone()
    )

    # C) build conditions and order
    conditions_by_patch = {}
    if condition_indices is not None:
        # assume same indices for all images
        for idx, val in zip(condition_indices, condition_values):
            r, c = divmod(idx, image_size)
            pr, pc = r // patch_size, c // patch_size
            pidx = pr * grid_size + pc
            local = (r % patch_size)*patch_size + (c % patch_size)
            conditions_by_patch.setdefault(pidx, []).append((local, val))

    observed = sorted(conditions_by_patch.keys())
    all_p = set(range(total_patches))
    unobs = list(all_p - set(observed))
    if generation_order == "manhattan" and observed:
        dist_list = []
        for p in unobs:
            pr, pc = divmod(p, grid_size)
            s = sum(abs(pr - (op//grid_size)) + abs(pc - (op%grid_size)) for op in observed)
            dist_list.append((p,s))
        dist_list.sort(key=lambda x:x[1])
        unobs = [p for p,_ in dist_list]
    elif generation_order=="raster":
        unobs.sort()
    elif generation_order=="random":
        np.random.shuffle(unobs)
    else:
        raise ValueError(generation_order)
    sampling_order = observed + unobs

    loglikes = [0.0]*batch_size
    for pidx in sampling_order:
        # 1) fast forward
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        cur_logits = logits[:, pidx, :].clone()   # [B, V]

        # 2) apply conditions mask
        if pidx in conditions_by_patch:
            valid = torch.ones(model.vocab_size, device=device, dtype=torch.bool)
            for local, val in conditions_by_patch[pidx]:
                valid &= (model.vocab[:, local].to(device)==val)
            cur_logits[:, ~valid] = -float('inf')

        # 3) sample per image
        probs = F.softmax(cur_logits, dim=-1)     # [B, V]
        samp  = torch.multinomial(probs, 1).squeeze(-1)  # [B]
        for i in range(batch_size):
            loglikes[i] += math.log(probs[i, samp[i]].item()+1e-10)

        # 4) write patches back
        gen_patches[:, pidx] = model.vocab[samp].to(device)

        # 5) re-init cache after the update
        model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # reconstruct
    imgs = [patches_to_image(gen_patches[i].cpu(), (image_size, image_size), patch_size)
            for i in range(batch_size)]
    return torch.stack(imgs), loglikes


def generate_images_batch_method1_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None
):
    model.eval()
    device = next(model.parameters()).device

    grid_size     = image_size // patch_size
    total_patches = grid_size * grid_size
    patch_dim     = patch_size * patch_size

    # A) init cache
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # B) init gen_patches
    gen_patches = (
        model.mask_token.to(device)
        .view(1, total_patches, patch_dim)
        .expand(batch_size, total_patches, patch_dim)
        .clone()
    )

    # C) prepare observed
    cond_by_patch = {}
    if condition_indices is not None:
        for idx, val in zip(condition_indices, condition_values):
            r, c = divmod(idx, image_size)
            pr, pc = r//patch_size, c//patch_size
            pidx = pr*grid_size + pc
            local = (r%patch_size)*patch_size + (c%patch_size)
            cond_by_patch.setdefault(pidx, []).append((local,val))
    observed = sorted(cond_by_patch.keys())

    # fill observed first
    loglikes = [0.0]*batch_size
    if observed:
        logits_obs, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        for p in observed:
            lc = logits_obs[:, p, :].clone()
            valid = torch.ones(model.vocab_size, device=device, dtype=torch.bool)
            for local,val in cond_by_patch[p]:
                valid &= (model.vocab[:, local].to(device)==val)
            lc[:, ~valid] = -float('inf')
            probs = F.softmax(lc, dim=-1)
            samp  = torch.multinomial(probs, 1).squeeze(-1)
            for i in range(batch_size):
                loglikes[i] += math.log(probs[i, samp[i]].item()+1e-10)
            gen_patches[:, p] = model.vocab[samp].to(device)
            model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # dynamic Method1
    to_fill = total_patches - len(observed)
    for _ in range(to_fill):
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)
        flat_logits = logits.view(batch_size*total_patches, -1)
        flat_probs  = F.softmax(flat_logits, dim=-1)
        flat_samps  = torch.multinomial(flat_probs,1).squeeze(-1)
        flat_lp     = flat_probs[torch.arange(batch_size*total_patches,device=device), flat_samps]

        B, P = batch_size, total_patches
        samps = flat_samps.view(B, P)
        pliks = flat_lp.view(B, P)
        # mask out observed
        for p in observed:
            pliks[:, p] = -float('inf')

        choice = torch.argmax(pliks, dim=1)  # [B]
        for i in range(B):
            loglikes[i] += math.log(pliks[i, choice[i]].item()+1e-10)

        gen_patches[torch.arange(B), choice] = model.vocab[samps[torch.arange(B),choice]].to(device)
        observed.extend(choice.tolist())
        model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # reconstruct
    imgs = [patches_to_image(gen_patches[i].cpu(), (image_size,image_size), patch_size)
            for i in range(batch_size)]
    return torch.stack(imgs), loglikes
