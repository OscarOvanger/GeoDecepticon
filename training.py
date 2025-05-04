from ViT_sampling import *
#from ViT_sampling_MLA import *
########################################
# Training Loop with Progress Bar and Updated Plotting
########################################

import wandb  # make sure wandb is installed: pip install wandb
def run_training(training_data, nr_epochs, batch_size, mask_rate, 
                 final_mask_rate, mask_schedule, patch_size, num_heads, 
                 num_layers, ffn_dim, learning_rate, emb_dim, vocab_cap,pretrained=False, 
                 use_pos_emb=True,pos_emb=None,use_rel_bias=True,rel_bias=None):
    """
    training_data: numpy array or tensor of shape [N, H, W] (binary images)
    nr_epochs: number of epochs for training
    batch_size: training batch size
    mask_rate: initial masking probability.
    final_mask_rate: final masking probability.
    mask_schedule: one of 'linear', 'exp', or 'log' that defines the growth of mask_rate.
    patch_size: patch size (square); for your experiment use 2 if desired.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize wandb
    wandb.init(project="ViT_conditional", config={
        "nr_epochs": nr_epochs,
        "batch_size": batch_size,
        "mask_rate": mask_rate,
        "final_mask_rate": final_mask_rate,
        "mask_schedule": mask_schedule,
        "patch_size": patch_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "ffn_dim": ffn_dim,
        "learning_rate": learning_rate,
        "emb_dim": emb_dim
    })

    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, dtype=torch.float32)

    dataset = BinaryImageDataset(training_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    B_total, H_img, W_img = training_data.shape
    num_patches = (H_img // patch_size) * (W_img // patch_size)
    patch_dim = patch_size * patch_size

    # Build vocabulary and mask token.
    vocab, counts, mask_token = build_vocabulary(training_data, patch_size, cap_size=vocab_cap)
    print("vocab size:\n", vocab.size(0))

    # Create the ViT model.
    if pretrained != False:
      model = pretrained
    else:
      model = StackedContextViT(
      vocab=vocab.to(device),
      mask_token=mask_token.to(device),
      patch_dim=patch_dim,
      num_patches=num_patches,
      emb_dim=emb_dim,
      num_heads=num_heads,
      num_layers=num_layers,
      ffn_dim=ffn_dim,
      use_pos_emb=use_pos_emb,
      pos_emb_init=pos_emb,     # either None or your [N×emb_dim] tensor
      use_rel_bias=use_rel_bias,
      rel_bias_init=rel_bias    # either None or your [N×N] tensor
      )
    model.to(device)
    model.vocab = model.vocab.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)
    criterion = nn.CrossEntropyLoss()

    #Weight for reconstruction loss
    alpha = 0.01
    
    for epoch in range(nr_epochs):
        # Compute the current masking rate according to the selected schedule.
        if mask_schedule == "linear":
            current_mask_rate = mask_rate + (final_mask_rate - mask_rate) * (epoch / (nr_epochs - 1))
        elif mask_schedule == "exp":
            current_mask_rate = mask_rate * ((final_mask_rate / mask_rate) ** (epoch / (nr_epochs - 1)))
        elif mask_schedule == "log":
            current_mask_rate = mask_rate + (final_mask_rate - mask_rate) * (math.log(epoch + 1) / math.log(nr_epochs))
        elif mask_schedule == "cyclic":
            current_mask_rate = mask_rate + (final_mask_rate - mask_rate) * ((epoch % 99) / 99)
        else:
            current_mask_rate = mask_rate  # fallback to constant rate
        

        model.train()
        total_loss = 0.0
        total_correct = 0
        total_masked = 0

        # Variables to store outputs from the last batch for plotting.
        last_batch_images = None
        last_batch_patches = None
        last_batch_mask = None
        last_batch_logits = None
        last_targets = None

        # Progress bar for this epoch.
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}/{nr_epochs}"):
            batch = batch.to(device)
            patches = BinaryImageDataset.batch_to_patches(batch, patch_size)  # shape [B, N, patch_dim]
            B, N, P = patches.shape

            logits, mask = model(patches, current_mask_rate)

            patches_bin = patches.round()
            patches_flat = patches_bin.view(B * N, P)
            dists = torch.cdist(patches_flat, model.vocab)
            targets = torch.argmin(dists, dim=1).view(B, N)

            if mask.sum() > 0:
                loss_mask = criterion(logits[mask], targets[mask])
            else:
                loss_mask = torch.tensor(0.0, device=device)

            pred_ids     = logits.argmax(dim=-1)             # [B, N]
            pred_patches = model.vocab[pred_ids]             # [B, N, D]
            # vectorized reassembly:
            recon_images = []
            for i in range(B):
                recon_images.append(
                    patches_to_image(pred_patches[i], (H_img, W_img), patch_size)
                )
            recon_images = torch.stack(recon_images).to(device)  # [B, H, W]
            loss_recon = F.l1_loss(recon_images, batch)

            # combined loss
            loss = loss_mask + alpha * loss_recon
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss_mask.item() * mask.sum().item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds[mask] == targets[mask]).sum().item()
            total_masked += mask.sum().item()

            # Save last batch data (including targets) for plotting.
            last_batch_images = batch.detach().cpu()
            last_batch_patches = patches.detach().cpu()
            last_batch_mask = mask.detach().cpu()
            last_batch_logits = logits.detach().cpu()
            last_targets = targets.detach().cpu()

        avg_loss = total_loss / max(1, total_masked)
        acc = total_correct / max(1, total_masked)
        scheduler.step() #Change laber
        # Log epoch metrics.
        wandb.log({
            "epoch": epoch+1,
            "masked_CE": loss_mask.item(),
            "recon_L1": loss_recon.item(),
            "combined_loss": loss.item(),
            "avg_loss": avg_loss,
            "masked_accuracy": acc,
            "mask_rate": current_mask_rate,
            "learning_rate": scheduler.get_last_lr()[0]
        })
        print(f"Epoch {epoch+1}/{nr_epochs} | Loss: {avg_loss:.4f} | Masked Acc: {acc:.4f}")

        if (epoch + 1) % 100 == 0:
          torch.save(model.state_dict(), f"model_epoch_{epoch+1}.pth")
          print(f"Saved model at epoch {epoch+1}")

        # ---- Every 10 epochs, perform conditional generation and log to wandb ----
        if (epoch+1) % 50 == 0:
            # Sample conditions for generation.
            condition_indices = np.array([1726,3797,4211,4543,3953,3347,1897,2015,4424,1942,2108,4350,1019,4068,2911])
            condition_values = np.array([0,1,0,1,1,1,0,1,0,1,0,1,1,1,0])
            condition_indices_x = (condition_indices // 80) - 8
            condition_indices_y = (condition_indices % 80) - 8
            flattened_condition_indices = np.zeros(len(condition_indices))
            for i in range(len(condition_indices)):
                flattened_condition_indices[i] = condition_indices_x[i] * 64 + condition_indices_y[i]
            flattened_condition_indices = flattened_condition_indices.astype(int)
            generated_img, log_likelihood_sum = generate_image(model, patch_size, W_img, condition_indices=flattened_condition_indices, condition_values=condition_values, generation_order="random")
            # Create a figure for the generated image.
            fig_gen, ax_gen = plt.subplots(figsize=(12,12))
            # Convert conditions to grid coordinates for plotting.
            ax_gen.imshow(generated_img.cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            ax_gen.scatter(condition_indices_y, condition_indices_x, c=condition_values, cmap='viridis')
            ax_gen.set_title(f"Conditional Generation @ Epoch {epoch+1}\nLL: {log_likelihood_sum:.4f}")
            ax_gen.axis("off")
            wandb.log({"conditional_generation": wandb.Image(fig_gen, caption=f"Epoch {epoch+1}")})
            plt.close(fig_gen)

        # ---- (Optional) Plotting the last batch reconstruction and log to wandb ----
        if (epoch+1) % 10 == 0:
          if last_batch_images is not None:
              last_image = last_batch_images[-1]         # [H_img, W_img]
              last_orig_patches = last_batch_patches[-1]    # [N, patch_dim]
              last_mask = last_batch_mask[-1]               # [N]
              last_logits = last_batch_logits[-1]           # [N, vocab_size]
              last_targets_img = last_targets[-1]           # [N]

              last_pred = last_logits.argmax(dim=-1)

              reconstructed_patches = last_orig_patches.clone()
              pred_patches = model.vocab[last_pred].detach().cpu()
              reconstructed_patches[last_mask] = pred_patches[last_mask]

              orig_img = last_image.numpy()
              recon_img = patches_to_image(reconstructed_patches, (H_img, W_img), patch_size).numpy()

              fig_recon, axes_recon = plt.subplots(1, 2, figsize=(24, 12))
              axes_recon[0].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
              axes_recon[0].set_title("Original Image")
              axes_recon[0].axis("off")

              axes_recon[1].imshow(recon_img, cmap='gray', vmin=0, vmax=1)
              axes_recon[1].set_title("Reconstructed Image")
              axes_recon[1].axis("off")

              num_patches_h = H_img // patch_size
              num_patches_w = W_img // patch_size
              N_patches = last_orig_patches.shape[0]
              for i in range(N_patches):
                  if last_mask[i]:
                      row = i // num_patches_w
                      col = i % num_patches_w
                      if last_pred[i] == last_targets_img[i]:
                          border_color = "green"
                      else:
                          border_color = "red"
                      rect = mpatches.Rectangle((col * patch_size -.5, row * patch_size -.5),
                                                patch_size, patch_size,
                                                linewidth=2, edgecolor=border_color, facecolor='none')
                      axes_recon[1].add_patch(rect)

              plt.suptitle(f"Epoch {epoch+1}")
              wandb.log({"reconstruction": wandb.Image(fig_recon, caption=f"Epoch {epoch+1}")})
              plt.close(fig_recon)

    return model, avg_loss
