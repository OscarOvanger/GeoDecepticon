import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
#import wandb
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
import time


########################################
# Helper: Convert Patches to Integer Code
########################################

def patch_to_int(patches):
    """
    Converts a tensor of patches (shape: [N, patch_dim]) containing binary values
    to an integer code. Assumes values are near 0 or 1.
    """
    patches_bin = patches.round().long()  # ensure binary
    patch_dim = patches_bin.shape[1]
    powers = (2 ** torch.arange(patch_dim - 1, -1, -1, device=patches.device)).unsqueeze(0)
    codes = (patches_bin * powers).sum(dim=1)
    return codes

def training_dat_to_patches(training_data, patch_size):
    """Extract patches from training data"""
    N, H, W = training_data.shape
    n_h, n_w = H // patch_size, W // patch_size
    
    # Reshape and extract patches using unfold
    x = training_data.reshape(N, 1, H, W)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).reshape(-1, patch_size * patch_size)

########################################
# Build enhanced vocabulary with learnable tokens
########################################

def build_vocabulary(training_data, patch_size, partial_masked_token=True):
    """Build vocabulary of discrete tokens including masked tokens"""
    # Extract patches
    patches = training_dat_to_patches(training_data, patch_size)
    
    # Binary patterns - include all possible binary configurations
    vocab = []
    for i in range(2**(patch_size**2)):
        binary = torch.tensor([int(bit) for bit in np.binary_repr(i, width=patch_size**2)], dtype=torch.float32)
        vocab.append(binary)
    
    # Add fully masked token
    fully_masked_token = torch.ones(patch_size**2) * 0.5
    vocab.append(fully_masked_token)
    fully_masked_idx = len(vocab) - 1
    
    # Add partially masked tokens
    partial_token_indices = []
    if partial_masked_token:
        for pos in range(patch_size**2):
            for val in range(2):
                token = torch.ones(patch_size**2) * 0.5
                token[pos] = float(val)
                vocab.append(token)
                partial_token_indices.append(len(vocab) - 1)
    
    # Create vocabulary tensor
    vocab = torch.stack(vocab)
    
    # Create trainable masked token
    masked_token_param = nn.Parameter(torch.mean(patches.float(), dim=0), requires_grad=True)
    
    # Create trainable partial masked tokens
    partial_masked_token_params = []
    if partial_masked_token:
        for pos in range(patch_size**2):
            for val in range(2):
                # Find relevant patches
                mask = patches[:, pos] == val
                if mask.any():
                    # Calculate mean of relevant patches
                    mean_patch = torch.mean(patches[mask].float(), dim=0)
                    param = nn.Parameter(mean_patch.clone(), requires_grad=True)
                else:
                    # Fallback
                    param = nn.Parameter(fully_masked_token.clone(), requires_grad=True)
                    param[pos] = float(val)
                
                partial_masked_token_params.append(param)
    
    return vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params

########################################
# Relative Position Self-Attention
########################################

class RelativePositionSelfAttention(nn.Module):
    """Self-attention with relative positional encoding"""
    def __init__(self, hidden_dim, num_heads, max_rel_dist=32, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.max_rel_dist = max_rel_dist
        
        # Relative position bias table
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_rel_dist + 1, num_heads))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate relative position bias
        rel_pos_idx = self._get_rel_pos_idx(seq_len).to(x.device)  # (seq_len, seq_len)
        rel_pos_bias = self.rel_pos_bias[rel_pos_idx]  # (seq_len, seq_len, num_heads)
        rel_pos_bias = rel_pos_bias.permute(2, 0, 1)  # (num_heads, seq_len, seq_len)
        
        # Apply attention with position bias
        attn_mask = torch.zeros(seq_len, seq_len, device=x.device)
        for h in range(rel_pos_bias.size(0)):
            attn_mask += rel_pos_bias[h]
        
        # Divide by number of heads to maintain scale
        attn_mask = attn_mask / rel_pos_bias.size(0)
        
        # Apply attention with the mask
        output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        
        return output
    
    def _get_rel_pos_idx(self, seq_len):
        """Generate relative position indices"""
        # Efficient vectorized computation
        i = torch.arange(seq_len).unsqueeze(1)
        j = torch.arange(seq_len).unsqueeze(0)
        rel_pos = i - j
        rel_pos = torch.clamp(rel_pos, -self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        return rel_pos

########################################
# Transformer Encoder Layer
########################################

class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer with relative positional encoding"""
    def __init__(self, hidden_dim, num_heads, ffn_dim, dropout=0.1, max_rel_dist=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = RelativePositionSelfAttention(hidden_dim, num_heads, max_rel_dist, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-LN architecture (more stable)
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

########################################
# Vision Transformer
########################################

class VisionTransformer(nn.Module):
    """Vision Transformer with discrete token vocabulary and relative positional encoding"""
    def __init__(self, vocab, fully_masked_idx, partial_token_indices, 
                 masked_token_param, partial_masked_token_params, patch_size, 
                 num_heads, num_layers, ffn_dim, hidden_dim, max_rel_dist=32, dropout=0.1):
        super().__init__()
        # Store vocabulary and token parameters
        self.register_buffer('vocab', vocab)
        self.fully_masked_idx = fully_masked_idx
        self.partial_token_indices = partial_token_indices
        self.masked_token_param = masked_token_param
        self.partial_masked_token_params = nn.ParameterList(partial_masked_token_params)
        self.mask_token = self.masked_token_param  # For compatibility with image generation code
        
        self.patch_size = patch_size
        self.vocab_size = vocab.shape[0]
        
        # Compute vocabulary integers for faster lookup
        self.vocab_int = patch_to_int(self.vocab)
        
        # Embedding layer
        self.token_embedding = nn.Linear(patch_size**2, hidden_dim)
        
        # Transformer encoder
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, max_rel_dist)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        # Initialize projection layers
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.zeros_(self.token_embedding.bias)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def _find_closest_token(self, patch):
        """Find closest vocabulary token using efficient vector operations"""
        distances = torch.sum((self.vocab - patch.unsqueeze(0))**2, dim=1)
        return torch.argmin(distances)
    
    def _process_batch(self, patches):
        """Process a batch of patches efficiently"""
        B, num_patches, patch_dim = patches.shape
        device = patches.device
        
        # Check for fully masked patches (all values close to 0.5)
        is_fully_masked = torch.all(torch.abs(patches - 0.5) < 0.1, dim=-1)  # (B, num_patches)
        
        # Prepare outputs
        processed_patches = torch.zeros_like(patches)
        token_indices = torch.zeros((B, num_patches), dtype=torch.long, device=device)
        
        # Process each batch item
        for b in range(B):
            for i in range(num_patches):
                patch = patches[b, i]
                
                if is_fully_masked[b, i]:
                    # Fully masked
                    token_indices[b, i] = self.fully_masked_idx
                    processed_patches[b, i] = self.masked_token_param
                else:
                    # Check for partial masking
                    mask_positions = torch.abs(patch - 0.5) < 0.1
                    binary_positions = ~mask_positions
                    
                    if torch.any(mask_positions) and torch.any(binary_positions):
                        # Partially masked - find first observed position
                        for pos in range(patch_dim):
                            if binary_positions[pos]:
                                val = round(patch[pos].item())
                                partial_idx = pos * 2 + val
                                if partial_idx < len(self.partial_token_indices):
                                    token_indices[b, i] = self.partial_token_indices[partial_idx]
                                    processed_patches[b, i] = self.partial_masked_token_params[partial_idx]
                                break
                    else:
                        # Regular binary token
                        idx = self._find_closest_token(patch)
                        token_indices[b, i] = idx
                        processed_patches[b, i] = patch
        
        return processed_patches, token_indices
    
    def forward(self, patches):
        """Forward pass"""
        # If input is just a single patch sequence (for generation), add batch dimension
        if len(patches.shape) == 2:
            patches = patches.unsqueeze(0)
            
        # Process patches and get token indices
        processed_patches, token_indices = self._process_batch(patches)
        
        # Generate embeddings
        embeddings = self.token_embedding(processed_patches)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer
        z = self.transformer(embeddings)
        
        # Project to vocabulary space
        logits = self.output_projection(z)
        
        return logits

########################################
# Dataset and Utility Functions
########################################

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        self.images = images  # shape: (N, H, W)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx]
    
    @staticmethod
    def batch_to_patches(images, patch_size):
        B, H, W = images.shape
        patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)
        return patches
    
    @staticmethod
    def patches_to_image(patches, patch_size, image_size):
        # patches: (num_patches, patch_dim) ordered in raster scan.
        image = torch.zeros(image_size, image_size, device=patches.device)
        idx = 0
        for i in range(0, image_size, patch_size):
            for j in range(0, image_size, patch_size):
                image[i:i+patch_size, j:j+patch_size] = patches[idx].view(patch_size, patch_size)
                idx += 1
        return image

########################################
# Reconstruction Function
########################################

def test_reconstruction(model, original_sample, masked_sample):
    """Reconstruct masked patches in an image"""
    H, W = original_sample.shape
    patch_size = model.patch_size
    n_h = H // patch_size
    n_w = W // patch_size
    
    # Extract patches from the masked sample
    patches = []
    mask_indices = []
    
    for i in range(n_h):
        for j in range(n_w):
            # Get the patch
            patch = masked_sample[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.reshape(-1))
            
            # Check if this patch was masked (contains values close to 0.5)
            was_masked = torch.any(torch.abs(patch - 0.5) < 0.1)
            if was_masked:
                mask_indices.append(len(patches) - 1)
    
    patches_tensor = torch.stack(patches).to(masked_sample.device)
    
    # Forward pass
    with torch.no_grad():
        logits = model(patches_tensor)
    
    # Get top predictions
    token_indices = torch.argmax(logits, dim=-1)  # (n_patches)
    
    # Create a copy of the original sample for reconstruction
    reconstructed = original_sample.clone()
    
    # Only replace the masked patches
    for idx in mask_indices:
        i = (idx // n_w) * patch_size
        j = (idx % n_w) * patch_size
        
        # Get the token from vocabulary
        token = model.vocab[token_indices[0][idx]].reshape(patch_size, patch_size)
        reconstructed[i:i+patch_size, j:j+patch_size] = token
    
    return reconstructed

########################################
# Image Generation (Conditional)
########################################

def sample_image_conditional(model, patch_size, image_size, temperature=1.0,
                              condition_indices=None, condition_values=None):
    """
    Generate one image using the model while enforcing conditions.
    Instead of a fixed raster-scan order, this function first generates the patches
    that contain condition values and then fills in the rest.

    Returns:
      - generated: a tensor of shape (1, num_patches, patch_dim)
      - log_likelihood: the sum of log-probabilities for the sampled tokens.
    """
    device = next(model.parameters()).device
    num_patches = (image_size // patch_size) ** 2
    patch_dim = patch_size * patch_size
    grid_size = image_size // patch_size

    # Build patch-level conditions dictionary
    patch_conditions = {}
    if condition_indices is not None and condition_values is not None:
        for cond_idx, cond_val in zip(condition_indices, condition_values):
            global_row = int(cond_idx) // image_size
            global_col = int(cond_idx) % image_size
            patch_row = global_row // patch_size
            patch_col = global_col // patch_size
            patch_index = patch_row * grid_size + patch_col
            local_row = global_row % patch_size
            local_col = global_col % patch_size
            local_index = local_row * patch_size + local_col
            if patch_index not in patch_conditions:
                patch_conditions[patch_index] = {}
            patch_conditions[patch_index][local_index] = float(cond_val)

    # Helper to compute patch coordinates
    def patch_coords(i):
        return (i // grid_size, i % grid_size)

    # Determine generation order
    # First, the conditioned patch indices
    conditioned_indices = set(patch_conditions.keys())
    conditioned_list = sorted(list(conditioned_indices))

    # Then, the remaining indices sorted by minimum Manhattan distance to any conditioned patch
    remaining = [i for i in range(num_patches) if i not in conditioned_indices]
    def min_distance(i):
        r_i, c_i = patch_coords(i)
        return min(abs(r_i - patch_coords(j)[0]) + abs(c_i - patch_coords(j)[1])
                   for j in conditioned_indices) if conditioned_indices else 0
    remaining = sorted(remaining, key=min_distance)

    # Final generation order
    order_list = conditioned_list + remaining

    # Initialize generated patches with the mask token
    generated = model.mask_token.detach().clone().unsqueeze(0).repeat(num_patches, 1)
    log_likelihood = 0.0
    generated = generated.unsqueeze(0)  # shape: (1, num_patches, patch_dim)

    # Generate patches in the determined order
    for idx in order_list:
        # Check if this patch has a condition
        cond = patch_conditions.get(idx, None)
        
        # Compute logits for the entire sequence
        logits = model(generated)  # shape: (1, num_patches, vocab_size)
        logits_i = logits[0, idx] / temperature
        
        # Apply conditional masking if needed
        if cond is not None:
            candidate_mask = torch.ones(model.vocab_size, dtype=torch.bool, device=logits_i.device)
            for local_idx, cond_val in cond.items():
                candidate_mask = candidate_mask & (model.vocab[:, local_idx] == cond_val)
            logits_i = logits_i.masked_fill(~candidate_mask, -1e9)
        
        # Sample from the distribution
        probs = torch.softmax(logits_i, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        log_prob = torch.log(probs[token] + 1e-10)
        log_likelihood += log_prob.item()
        
        # Update the generated sequence
        patch = model.vocab[token]
        generated[0, idx] = patch

    return generated, log_likelihood

########################################
# Training Function
########################################

def train_vit(model, train_data, test_data=None, batch_size=32, num_epochs=100, 
              min_mask_ratio=0.05, max_mask_ratio=0.5, cycle_length=10, 
              use_wandb=True, save_dir='./checkpoints', visualize_every=10):
    """Train Vision Transformer with cyclical mask ratio"""
    device = next(model.parameters()).device
    image_size = train_data.shape[1]
    
    # Create directories
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Select test samples for quick evaluation
    if test_data is None:
        test_data = train_data
    test_indices = np.random.choice(len(test_data), min(3, len(test_data)), replace=False)
    test_samples = test_data[test_indices].to(device)
    
    # Setup optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Create dataset and dataloader
    dataset = BinaryImageDataset(train_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Training metrics
    cycle_losses = []
    cycle_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Get mask ratio for this epoch using cyclical schedule
        cycle = epoch // cycle_length
        epoch_in_cycle = epoch % cycle_length
        ratio_range = max_mask_ratio - min_mask_ratio
        mask_ratio = min_mask_ratio + ratio_range * (epoch_in_cycle / (cycle_length - 1))
        mask_ratio = np.clip(mask_ratio + np.random.uniform(-0.05, 0.05), min_mask_ratio, max_mask_ratio)
        
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Process batches with progress bar
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch in pbar:
                batch = batch.to(device)
                B = batch.shape[0]
                
                # Convert images to patches
                patches = BinaryImageDataset.batch_to_patches(batch, model.patch_size)
                num_patches = patches.shape[1]
                patch_dim = patches.shape[2]
                
                # Create mask for patches (which will be masked)
                mask = torch.rand(B, num_patches, device=device) < mask_ratio
                
                # Choose some patches for partial masking
                partial_mask_ratio = 0.3  # 30% of masked patches will be partially masked
                partial = (torch.rand(B, num_patches, device=device) < partial_mask_ratio) & mask
                full = mask & (~partial)
                
                # Create the masked input
                masked_patches = patches.clone()
                
                # Apply full masking
                if full.any():
                    masked_patches[full] = model.mask_token
                
                # Apply partial masking
                if partial.any():
                    partial_idx = torch.nonzero(partial)
                    num_partial = partial_idx.shape[0]
                    new_patches = torch.full((num_partial, patch_dim), 0.5, device=device)
                    rand_positions = torch.randint(0, patch_dim, (num_partial,), device=device)
                    orig_vals = patches[partial_idx[:, 0], partial_idx[:, 1], :].gather(1, rand_positions.unsqueeze(1)).squeeze(1)
                    new_patches[torch.arange(num_partial), rand_positions] = orig_vals
                    masked_patches[partial_idx[:, 0], partial_idx[:, 1]] = new_patches
                
                # Compute targets (find the closest token index for each original patch)
                targets = torch.zeros(B, num_patches, dtype=torch.long, device=device)
                for b in range(B):
                    for p in range(num_patches):
                        targets[b, p] = model._find_closest_token(patches[b, p])
                
                # Forward pass
                optimizer.zero_grad()
                logits = model(masked_patches)
                
                # Only compute loss on masked tokens
                if mask.sum() > 0:
                    loss = criterion(logits[mask], targets[mask])
                else:
                    loss = torch.tensor(0.0, device=device)
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                batch_correct = (predictions[mask] == targets[mask]).sum().item()
                batch_total = mask.sum().item()
                
                correct += batch_correct
                total += batch_total
                total_loss += loss.item() * B
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{batch_correct/max(1, batch_total):.4f}",
                    'mask': f"{mask_ratio:.2f}"
                })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(dataset)
        avg_acc = correct / max(1, total)
        
        # Store metrics
        cycle_losses.append(avg_loss)
        cycle_accs.append(avg_acc)
        
        # Log epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Mask: {mask_ratio:.2f}")
        
        # Log to wandb
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'loss': avg_loss,
                'accuracy': avg_acc,
                'mask_ratio': mask_ratio,
            })
        
        # End of cycle or visualization epoch
        if (epoch + 1) % cycle_length == 0 or (epoch + 1) % visualize_every == 0 or (epoch + 1) == num_epochs:
            # Evaluate on test samples
            model.eval()
            with torch.no_grad():
                # Show reconstruction of test samples
                for i, sample in enumerate(test_samples):
                    # Create masked version with current mask ratio
                    masked_sample = sample.clone()
                    H, W = sample.shape
                    n_h, n_w = H // model.patch_size, W // model.patch_size
                    total_patches = n_h * n_w
                    
                    # Mask random patches
                    mask_indices = np.random.choice(
                        total_patches, 
                        int(mask_ratio * total_patches), 
                        replace=False
                    )
                    
                    for idx in mask_indices:
                        i_patch = idx // n_w
                        j_patch = idx % n_w
                        masked_sample[i_patch*model.patch_size:(i_patch+1)*model.patch_size, 
                                     j_patch*model.patch_size:(j_patch+1)*model.patch_size] = 0.5
                    
                    # Reconstruct
                    reconstructed = test_reconstruction(model, sample, masked_sample)
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                    
                    axes[0].imshow(sample.cpu().numpy(), cmap='gray')
                    axes[0].set_title("Original")
                    axes[0].axis('off')
                    
                    axes[1].imshow(masked_sample.cpu().numpy(), cmap='gray')
                    axes[1].set_title(f"Masked ({mask_ratio:.2f})")
                    axes[1].axis('off')
                    
                    axes[2].imshow(reconstructed.cpu().numpy(), cmap='gray')
                    axes[2].set_title("Reconstructed")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Log to wandb
                    if use_wandb:
                        wandb.log({f"reconstruction_{i}": wandb.Image(fig)})
                    
                    plt.close(fig)
                
                # Generate conditional sample
                if (epoch + 1) % cycle_length == 0 or (epoch + 1) == num_epochs:
                    # Create random conditions (pixel indices and values)
                    num_conditions = 30
                    condition_indices = np.random.choice(image_size * image_size, num_conditions, replace=False)
                    condition_values = np.random.choice([0, 1], num_conditions)
                    
                    # Generate conditional image
                    gen, ll = sample_image_conditional(
                        model, model.patch_size, image_size, temperature=1.0,
                        condition_indices=condition_indices, condition_values=condition_values
                    )
                    
                    # Convert to image
                    gen_img = BinaryImageDataset.patches_to_image(gen[0], model.patch_size, image_size)
                    
                    # Visualize
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(gen_img.cpu().numpy(), cmap='gray')
                    ax.set_title(f"Conditional Sample (LL: {ll:.2f})")
                    ax.axis('off')
                    
                    # Log to wandb
                    if use_wandb:
                        wandb.log({"conditional_sample": wandb.Image(fig)})
                    
                    plt.close(fig)
            
            # Save checkpoint
            torch.save(model.state_dict(), f"{save_dir}/vit_patch{model.patch_size}_epoch{epoch+1}.pt")
        
        # End of cycle
        if (epoch + 1) % cycle_length == 0 or (epoch + 1) == num_epochs:
            cycle_avg_loss = np.mean(cycle_losses)
            cycle_avg_acc = np.mean(cycle_accs)
            
            print(f"Cycle {cycle+1} completed, Avg Loss: {cycle_avg_loss:.4f}, Avg Acc: {cycle_avg_acc:.4f}")
            
            if use_wandb:
                wandb.log({
                    'cycle': cycle + 1,
                    'cycle_avg_loss': cycle_avg_loss,
                    'cycle_avg_acc': cycle_avg_acc,
                })
            
            # Reset cycle metrics
            cycle_losses = []
            cycle_accs = []
    
    return model

########################################
# Main Training Pipeline
########################################

def run_training(data_array, patch_size=3, hidden_dim=128, num_heads=8, num_layers=6, 
                ffn_dim=512, dropout=0.1, batch_size=32, num_epochs=100, 
                min_mask_ratio=0.05, max_mask_ratio=0.5, cycle_length=10, 
                use_wandb=True, wandb_project="vit-discrete-tokens", wandb_name=None,
                save_dir='./checkpoints', test_split=0.1, seed=42):
    """Complete pipeline for training Vision Transformer with discrete tokens"""
    # Set random seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process data
    if isinstance(data_array, torch.Tensor):
        data_array = data_array.numpy()
    
    # Split data
    n_samples = data_array.shape[0]
    indices = np.random.permutation(n_samples)
    n_test = int(n_samples * test_split)
    
    train_data = torch.tensor(data_array[indices[n_test:]], dtype=torch.float32)
    test_data = torch.tensor(data_array[indices[:n_test]], dtype=torch.float32)
    
    # Normalize if needed
    if train_data.max() > 1.0:
        train_data /= 255.0
        test_data /= 255.0
    
    # Initialize wandb if needed
    if use_wandb:
        run_name = wandb_name or f"vit_p{patch_size}_h{hidden_dim}_l{num_layers}_{int(time.time())}"
        wandb.init(
            project=wandb_project,
            name=run_name,
            config={
                "patch_size": patch_size,
                "hidden_dim": hidden_dim,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "ffn_dim": ffn_dim,
                "dropout": dropout,
                "batch_size": batch_size,
                "min_mask_ratio": min_mask_ratio,
                "max_mask_ratio": max_mask_ratio,
                "cycle_length": cycle_length
            }
        )
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params = build_vocabulary(
        train_data, patch_size, partial_masked_token=True
    )
    
    # Create model
    print(f"Creating model with {hidden_dim} hidden dim, {num_layers} layers, {num_heads} heads")
    model = VisionTransformer(
        vocab=vocab,
        fully_masked_idx=fully_masked_idx,
        partial_token_indices=partial_token_indices,
        masked_token_param=masked_token_param,
        partial_masked_token_params=partial_masked_token_params,
        patch_size=patch_size,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    ).to(device)
    
    # Train model
    print("Starting training...")
    model = train_vit(
        model=model,
        train_data=train_data,
        test_data=test_data,
        batch_size=batch_size,
        num_epochs=num_epochs,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        cycle_length=cycle_length,
        use_wandb=use_wandb,
        save_dir=save_dir
    )
    
    if use_wandb:
        wandb.finish()
    
    return model
