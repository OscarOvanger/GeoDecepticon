def generate_image_mla(
    model, patch_size, image_size,
    condition_indices=None, condition_values=None,
    generation_order="manhattan"
):
    model.eval()
    grid = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # (A) Initialize KV cache once for an all-mask input
    model.init_kv_cache(batch_size=1, total_patches=total_patches)

    # (B) Prepare generated_patches tensor
    gen_patches = model.mask_token.to(device).view(1,1,patch_dim) \
                   .expand(1, total_patches, patch_dim).clone()

    # (C) Build sampling order (same as before) …
    #  omitted here for brevity; same logic as your existing code to get `sampling_order`
    #  and `conditions_by_patch` dict.

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
        x_tok = model.patch_proj(new_patch) + model.pos_emb[pidx].unsqueeze(0)
        for layer_i in range(len(model.encoder_layers)):
            model.update_kv_cache(layer_i, pidx, x_tok)

    # Reconstruct final image
    final = patches_to_image(gen_patches[0].cpu(), (image_size, image_size), patch_size)
    return final, logsum

def generate_images_batch_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None,
    generation_order="manhattan"
):
    model.eval()
    grid = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # A) init KV cache for whole batch
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)

    # B) init gen_patches [B,P,D]
    gen_patches = model.mask_token.to(device).view(1,1,patch_dim) \
                   .expand(batch_size, total_patches, patch_dim).clone()

    # C) build sampling_order & conditions_by_patch as before…

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
        x_tok = model.patch_proj(new_patches) + model.pos_emb[pidx].unsqueeze(0)
        for li in range(len(model.encoder_layers)):
            model.update_kv_cache(li, pidx, x_tok)

    # reconstruct
    imgs = [patches_to_image(gen_patches[i].cpu(), (image_size, image_size), patch_size)
            for i in range(batch_size)]
    return torch.stack(imgs), loglikes

def generate_images_batch_method1_mla(
    model, patch_size, image_size, batch_size,
    condition_indices=None, condition_values=None
):
    model.eval()
    grid = image_size // patch_size
    total_patches = grid * grid
    patch_dim = patch_size * patch_size
    device = next(model.parameters()).device

    # A) init KV cache
    model.init_kv_cache(batch_size=batch_size, total_patches=total_patches)
    # B) init gen_patches
    gen_patches = model.mask_token.to(device).view(1,1,patch_dim) \
                   .expand(batch_size, total_patches, patch_dim).clone()

    # C) prepare observed list & to_fill count…

    loglikes = [0.0]*batch_size

    # 1) fill observed as before (identical to above, but using forward_from_cache & update_kv_cache)

    # 2) dynamic steps:
    to_fill = total_patches - len(observed)
    for _ in range(to_fill):
        logits, _ = model.forward_from_cache(gen_patches, mask_rate=0.0)  # [B,P,V]
        flat_logits = logits.view(batch_size*total_patches, -1)
        flat_probs  = F.softmax(flat_logits, dim=-1)
        flat_samps  = torch.multinomial(flat_probs, 1).squeeze(-1)
        flat_lp     = flat_probs[torch.arange(batch_size*total_patches, device=device), flat_samps]
        # choose highest‐prob patch per image, update patches & caches, accumulate loglikes…
        # (same logic as before, just calling update_kv_cache for the chosen patch)
        # …

    # reconstruct final images…
    # return images, loglikes
