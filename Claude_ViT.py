import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import wandb
from tqdm.auto import tqdm
import random
import time

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

# Utility functions for patches
def training_dat_to_patches(training_data, patch_size):
    """Extract patches from training data using unfold"""
    N, H, W = training_data.shape
    
    # Calculate number of patches in each dimension
    n_h = H // patch_size
    n_w = W // patch_size
    
    # Reshape for unfold operation
    x = training_data.reshape(N, 1, H, W)
    
    # Extract patches using unfold
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    
    # Reshape to get final shape (N*n_h*n_w, patch_size*patch_size)
    patches = patches.transpose(1, 2).reshape(-1, patch_size * patch_size)
    
    return patches

def build_vocabulary(training_data, patch_size, partial_masked_token=True):
    """Build vocabulary of discrete tokens including masked tokens"""
    # Extract patches
    patches = training_dat_to_patches(training_data, patch_size)
    
    # Binary patterns
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

# Visualize vocabulary tokens
def visualize_vocabulary(model, num_tokens=10):
    """Visualize sample tokens from vocabulary"""
    # Choose a mix of tokens
    token_indices = [0]  # Always include the all-zeros token
    token_indices.extend([
        model.fully_masked_idx,  # The fully masked token
        model.partial_token_indices[0],  # First partial token
        2**(model.patch_size**2) - 1  # All ones token
    ])
    
    # Add some random tokens
    remaining_count = num_tokens - len(token_indices)
    if remaining_count > 0:
        random_indices = torch.randint(0, model.vocab_size, (remaining_count,))
        token_indices.extend(random_indices.tolist())
    
    # Plot tokens
    fig, axs = plt.subplots(1, len(token_indices), figsize=(len(token_indices) * 2, 2))
    
    for i, idx in enumerate(token_indices):
        token = model.vocab[idx].reshape(model.patch_size, model.patch_size).cpu().numpy()
        axs[i].imshow(token, cmap='gray', vmin=0, vmax=1)
        axs[i].set_title(f"Token {idx}")
        axs[i].axis('off')
    
    plt.tight_layout()
    return fig

# Model Components
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
        
        # Use PyTorch's MultiheadAttention with additional attention bias
        attn_mask = torch.zeros(seq_len, seq_len, device=x.device)
        # Add relative position bias to attention mask
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

class VisionTransformer(nn.Module):
    """Vision Transformer with discrete token vocabulary and relative positional encoding"""
    def __init__(self, vocab, fully_masked_idx, partial_token_indices, 
                 masked_token_param, partial_masked_token_params, patch_size, 
                 num_heads, num_layers, ffn_dim, hidden_dim, max_rel_dist=32, dropout=0.0):
        super().__init__()
        # Store vocabulary and token parameters
        self.register_buffer('vocab', vocab)
        self.fully_masked_idx = fully_masked_idx
        self.partial_token_indices = partial_token_indices
        self.masked_token_param = masked_token_param
        self.partial_masked_token_params = nn.ParameterList(partial_masked_token_params)
        
        self.patch_size = patch_size
        self.vocab_size = vocab.shape[0]
        
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
        # Process patches and get token indices
        processed_patches, token_indices = self._process_batch(patches)
        
        # Generate embeddings
        embeddings = self.token_embedding(processed_patches)
        embeddings = self.dropout(embeddings)
        
        # Apply transformer
        z = self.transformer(embeddings)
        
        # Project to vocabulary space
        logits = self.output_projection(z)
        
        return logits, token_indices
    
    def predict_patches(self, logits):
        """Convert logits to patches"""
        token_indices = torch.argmax(logits, dim=-1)  # (B, num_patches)
        patches = self.vocab[token_indices]
        return patches

# Training and visualization functions
def test_reconstruction(model, original_sample, masked_sample, patch_size):
    """
    Reconstruction function that preserves unmasked patches
    
    Args:
        model: The Vision Transformer model
        original_sample: The original unmasked image
        masked_sample: The image with masked patches (values of 0.5)
        patch_size: Size of each patch
    """
    H, W = original_sample.shape
    n_h = H // patch_size
    n_w = W // patch_size
    
    # Extract patches from the masked sample
    patches = []
    mask_indices = []  # Track which patches were masked
    
    for i in range(n_h):
        for j in range(n_w):
            # Get the patch
            patch = masked_sample[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.reshape(-1))
            
            # Check if this patch was masked (contains values close to 0.5)
            was_masked = torch.any(torch.abs(patch - 0.5) < 0.1)
            if was_masked:
                mask_indices.append(len(patches) - 1)
    
    patches_tensor = torch.stack(patches).unsqueeze(0).to(masked_sample.device)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(patches_tensor)
    
    # Get top predictions
    token_indices = torch.argmax(logits, dim=-1)[0]  # (n_patches)
    
    # Create a copy of the original sample for reconstruction
    reconstructed = original_sample.clone()
    
    # Only replace the masked patches
    for idx in mask_indices:
        i = (idx // n_w) * patch_size
        j = (idx % n_w) * patch_size
        
        # Get the token from vocabulary
        token = model.vocab[token_indices[idx]].reshape(patch_size, patch_size)
        reconstructed[i:i+patch_size, j:j+patch_size] = token
    
    return reconstructed, token_indices, mask_indices

def calculate_accuracy(model, batch_patches, masked_patches, mask_indices, targets):
    """Calculate accuracy on masked patches"""
    device = next(model.parameters()).device
    batch_size = batch_patches.shape[0]
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(masked_patches)
    
    # Get predictions
    predictions = torch.argmax(logits, dim=-1)  # (B, num_patches)
    
    # Calculate accuracy on masked patches only
    correct = 0
    total = 0
    
    for b in range(batch_size):
        for idx in mask_indices[b]:
            correct += (predictions[b, idx] == targets[b, idx]).item()
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def create_reconstruction_grid(original_samples, masked_samples, reconstructed_samples, epoch, mask_ratio, loss):
    """Create grid of images for visualization with better formatting"""
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    num_samples = len(original_samples)
    fig = plt.figure(figsize=(15, 4 * num_samples))
    gs = GridSpec(num_samples, 3, figure=fig)
    
    for i in range(num_samples):
        # Plot original
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(original_samples[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Original")
        ax.axis('off')
        
        # Plot masked
        ax = fig.add_subplot(gs[i, 1])
        ax.imshow(masked_samples[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Masked (ratio: {mask_ratio:.2f})")
        ax.axis('off')
        
        # Plot reconstructed
        ax = fig.add_subplot(gs[i, 2])
        ax.imshow(reconstructed_samples[i].cpu().numpy(), cmap='gray', vmin=0, vmax=1)
        ax.set_title(f"Reconstructed")
        ax.axis('off')
    
    # Add epoch information
    plt.suptitle(f"Epoch {epoch}, Loss: {loss:.4f}", fontsize=16)
    plt.tight_layout()
    
    return fig

def train_vit_with_cyclical_masking(
    model, 
    training_data, 
    optimizer, 
    criterion, 
    batch_size=32, 
    num_epochs=100, 
    min_mask_ratio=0.05, 
    max_mask_ratio=0.5, 
    cycle_length=10,
    log_every=10,
    use_wandb=True,
    save_dir='./model_checkpoints',
    visualize_every=5
):
    """
    Training with cyclical mask ratio and improved monitoring/visualization
    
    Args:
        model: The Vision Transformer model
        training_data: Training data tensor
        optimizer: Optimizer
        criterion: Loss function
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        min_mask_ratio: Minimum masking ratio
        max_mask_ratio: Maximum masking ratio
        cycle_length: Length of mask ratio cycle (in epochs)
        log_every: Log metrics every N batches
        use_wandb: Whether to use Weights & Biases for logging
        save_dir: Directory to save model checkpoints
        visualize_every: Create visualizations every N epochs
    """
    device = next(model.parameters()).device
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/reconstructions", exist_ok=True)
    
    # Select test samples for visualization
    test_indices = np.random.choice(len(training_data), 5, replace=False)
    test_samples = training_data[test_indices].to(device)
    
    # Setup for tracking metrics
    cycle_metrics = {
        'losses': [],
        'accuracies': [],
        'mask_ratios': [],
    }
    
    # Track all epoch metrics for better visualization
    all_epoch_losses = []
    all_epoch_accs = []
    all_epoch_mask_ratios = []
    
    # Overall training stats
    all_cycle_avg_losses = []
    all_cycle_avg_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Get mask ratio for this epoch using cyclical schedule
        mask_ratio, cycle, epoch_in_cycle = get_mask_ratio_for_epoch(
            epoch, min_mask_ratio, max_mask_ratio, cycle_length
        )
        
        # Print using tqdm for better visibility
        print(f"Epoch {epoch+1}/{num_epochs}, Cycle {cycle+1}, Epoch in cycle {epoch_in_cycle+1}, Mask ratio: {mask_ratio:.3f}")
        
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Extract image dimensions
        H, W = training_data.shape[1:]
        n_h = H // model.patch_size
        n_w = W // model.patch_size
        patches_per_image = n_h * n_w
        
        # Create data loader
        num_batches = (len(training_data) + batch_size - 1) // batch_size
        
        # Process in batches with progress bar
        with tqdm(total=num_batches, desc=f"Training Epoch {epoch+1}/{num_epochs}", 
                 postfix={'loss': 0.0, 'acc': 0.0}, leave=True) as batch_pbar:
            for batch_idx in range(num_batches):
                # Get batch of images
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(training_data))
                batch_size_actual = end_idx - start_idx
                
                # Get images for this batch
                batch_images = training_data[start_idx:end_idx].to(device)
                
                # Extract patches
                batch_patches = []
                for img in batch_images:
                    img_patches = []
                    for i in range(0, H, model.patch_size):
                        for j in range(0, W, model.patch_size):
                            if i + model.patch_size <= H and j + model.patch_size <= W:
                                patch = img[i:i+model.patch_size, j:j+model.patch_size].reshape(-1)
                                img_patches.append(patch)
                    batch_patches.append(torch.stack(img_patches))
                
                batch_patches = torch.stack(batch_patches)
                
                # Create masked version with current mask ratio
                masked_patches = batch_patches.clone()
                mask_indices = []
                
                for b in range(batch_size_actual):
                    # Random indices to mask
                    num_to_mask = int(patches_per_image * mask_ratio)
                    indices = torch.randperm(patches_per_image)[:num_to_mask]
                    mask_indices.append(indices)
                    
                    for idx in indices:
                        r = torch.rand(1).item()
                        if r < 0.2:  # 20% fully masked
                            masked_patches[b, idx] = 0.5
                        elif r < 0.4:  # 20% partially masked
                            pos = torch.randint(0, model.patch_size**2, (1,)).item()
                            val = batch_patches[b, idx, pos].item()
                            masked_patches[b, idx] = 0.5
                            masked_patches[b, idx, pos] = val
                        # 60% unchanged for faster learning
                
                # Compute targets - find the corresponding vocabulary token
                targets = torch.zeros(batch_size_actual, patches_per_image, dtype=torch.long, device=device)
                for b in range(batch_size_actual):
                    for p in range(patches_per_image):
                        targets[b, p] = model._find_closest_token(batch_patches[b, p])
                
                # Forward pass
                optimizer.zero_grad()
                logits, _ = model(masked_patches)
                
                # Calculate loss
                loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
                
                # Backward and optimize
                loss.backward()
                
                # Clip gradients to prevent instability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                
                # Calculate accuracy on masked tokens
                batch_acc, correct, total = calculate_accuracy(
                    model, batch_patches, masked_patches, mask_indices, targets
                )
                
                # Update metrics
                epoch_loss += loss.item() * batch_size_actual
                epoch_correct += correct
                epoch_total += total
                
                # Update progress bar
                batch_pbar.set_postfix({
                    'loss': f"{loss.item():.4f}", 
                    'acc': f"{batch_acc:.4f}"
                })
                batch_pbar.update(1)
                
                # Log to W&B - only log key metrics
                if use_wandb and batch_idx % log_every == 0:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_accuracy': batch_acc,
                        'mask_ratio': mask_ratio,
                    })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(training_data)
        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        # Store metrics for this epoch
        cycle_metrics['losses'].append(avg_loss)
        cycle_metrics['accuracies'].append(avg_acc)
        cycle_metrics['mask_ratios'].append(mask_ratio)
        
        # Store for overall plots
        all_epoch_losses.append(avg_loss)
        all_epoch_accs.append(avg_acc)
        all_epoch_mask_ratios.append(mask_ratio)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Mask Ratio: {mask_ratio:.3f}")
        
        # Log epoch metrics to W&B - keep it focused on useful metrics
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'epoch_loss': avg_loss,
                'epoch_accuracy': avg_acc,
            })
        
        # If this is the end of a cycle, calculate and log cycle metrics
        if (epoch + 1) % cycle_length == 0 or (epoch + 1) == num_epochs:
            cycle_avg_loss = np.mean(cycle_metrics['losses'])
            cycle_avg_acc = np.mean(cycle_metrics['accuracies'])
            
            # Store cycle averages
            all_cycle_avg_losses.append(cycle_avg_loss)
            all_cycle_avg_accs.append(cycle_avg_acc)
            
            print(f"Cycle {cycle+1} completed, Avg Loss: {cycle_avg_loss:.4f}, Avg Acc: {cycle_avg_acc:.4f}")
            
            # Create and log cycle metrics plot
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            # Plot epoch losses for this cycle
            axes[0].plot(range(epoch - len(cycle_metrics['losses']) + 1, epoch + 1), cycle_metrics['losses'], 'r-o')
            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('Loss')
            axes[0].set_title(f'Cycle {cycle+1} Losses')
            axes[0].grid(True)
            
            # Plot epoch accuracies for this cycle
            axes[1].plot(range(epoch - len(cycle_metrics['accuracies']) + 1, epoch + 1), cycle_metrics['accuracies'], 'b-o')
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title(f'Cycle {cycle+1} Accuracies')
            axes[1].grid(True)
            
            # Plot mask ratios for this cycle
            axes[2].plot(range(epoch - len(cycle_metrics['mask_ratios']) + 1, epoch + 1), cycle_metrics['mask_ratios'], 'g-o')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Mask Ratio')
            axes[2].set_title(f'Cycle {cycle+1} Mask Ratios')
            axes[2].grid(True)
            
            plt.tight_layout()
            plt.savefig(f"{save_dir}/cycle_{cycle+1}_metrics.png")
            
            if use_wandb:
                wandb.log({
                    'cycle_completed': cycle + 1,
                    'cycle_avg_loss': cycle_avg_loss,
                    'cycle_avg_accuracy': cycle_avg_acc,
                    'cycle_metrics': wandb.Image(fig)
                })
            
            plt.close(fig)
            
            # Plot overall cycle metrics
            if len(all_cycle_avg_losses) > 1:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot cycle average losses
                axes[0].plot(range(1, len(all_cycle_avg_losses) + 1), all_cycle_avg_losses, 'r-o')
                axes[0].set_xlabel('Cycle')
                axes[0].set_ylabel('Average Loss')
                axes[0].set_title('Cycle Average Losses')
                axes[0].grid(True)
                
                # Plot cycle average accuracies
                axes[1].plot(range(1, len(all_cycle_avg_accs) + 1), all_cycle_avg_accs, 'b-o')
                axes[1].set_xlabel('Cycle')
                axes[1].set_ylabel('Average Accuracy')
                axes[1].set_title('Cycle Average Accuracies')
                axes[1].grid(True)
                
                plt.tight_layout()
                plt.savefig(f"{save_dir}/overall_cycle_metrics.png")
                
                if use_wandb:
                    wandb.log({
                        'overall_cycle_metrics': wandb.Image(fig)
                    })
                
                plt.close(fig)
            
            # Save model checkpoint at end of cycle
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'cycle': cycle,
                'cycle_avg_loss': cycle_avg_loss,
                'cycle_avg_acc': cycle_avg_acc
            }, f"{save_dir}/model_cycle_{cycle+1}.pt")
            
            # Reset cycle metrics
            cycle_metrics = {
                'losses': [],
                'accuracies': [],
                'mask_ratios': [],
            }
        
        # Generate visualizations periodically
        if epoch % visualize_every == 0 or (epoch + 1) == num_epochs:
            model.eval()
            with torch.no_grad():
                # Create masked versions and reconstruct
                masked_test_samples = []
                reconstructed_test_samples = []
                
                for sample in test_samples:
                    # Create a masked version
                    masked_sample = sample.clone()
                    H, W = sample.shape
                    n_h = H // model.patch_size
                    n_w = W // model.patch_size
                    total_patches = n_h * n_w
                    
                    # Use current epoch's mask ratio
                    mask_indices = np.random.choice(
                        total_patches, 
                        int(mask_ratio * total_patches), 
                        replace=False
                    )
                    
                    # Apply masking
                    for idx in mask_indices:
                        i_patch = idx // n_w
                        j_patch = idx % n_w
                        masked_sample[i_patch*model.patch_size:(i_patch+1)*model.patch_size, 
                                      j_patch*model.patch_size:(j_patch+1)*model.patch_size] = 0.5
                    
                    masked_test_samples.append(masked_sample)
                    
                    # Reconstruct
                    reconstructed, _, _ = test_reconstruction(
                        model, sample, masked_sample, model.patch_size
                    )
                    reconstructed_test_samples.append(reconstructed)
                
                # Create visualization grid
                fig = create_reconstruction_grid(
                    test_samples, 
                    masked_test_samples, 
                    reconstructed_test_samples, 
                    epoch + 1, 
                    mask_ratio, 
                    avg_loss
                )
                
                # Save the reconstruction grid locally by epoch number
                plt.savefig(f"{save_dir}/reconstructions/epoch_{epoch+1}.png")
                
                # Log visualization to W&B
                if use_wandb:
                    wandb.log({
                        "reconstruction_grid": wandb.Image(fig),
                        "epoch": epoch + 1
                    })
                
                # Close figure to free memory
                plt.close(fig)
    
    # Plot final overall metrics
    if len(all_epoch_losses) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot all epoch losses
        axes[0, 0].plot(range(1, len(all_epoch_losses) + 1), all_epoch_losses, 'r-')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Epoch Losses')
        axes[0, 0].grid(True)
        
        # Plot all epoch accuracies
        axes[0, 1].plot(range(1, len(all_epoch_accs) + 1), all_epoch_accs, 'b-')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Epoch Accuracies')
        axes[0, 1].grid(True)
        
        # Plot all mask ratios
        axes[1, 0].plot(range(1, len(all_epoch_mask_ratios) + 1), all_epoch_mask_ratios, 'g-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Mask Ratio')
        axes[1, 0].set_title('Mask Ratios')
        axes[1, 0].grid(True)
        
        # Plot cycle average losses
        axes[1, 1].plot(range(1, len(all_cycle_avg_losses) + 1), all_cycle_avg_losses, 'm-o')
        axes[1, 1].set_xlabel('Cycle')
        axes[1, 1].set_ylabel('Average Loss')
        axes[1, 1].set_title('Cycle Average Losses')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/final_metrics.png")
        
        if use_wandb:
            wandb.log({
                'final_metrics': wandb.Image(fig)
            })
        
        plt.close(fig)
    
    # Save final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'final_loss': avg_loss,
        'final_acc': avg_acc
    }, f"{save_dir}/model_final.pt")
    
    # Create a small helper HTML file to view reconstructions across epochs
    create_reconstruction_viewer(save_dir, visualize_every, num_epochs)
    
    return model, {
        'cycle_avg_losses': all_cycle_avg_losses,
        'cycle_avg_accs': all_cycle_avg_accs,
        'all_epoch_losses': all_epoch_losses,
        'all_epoch_accs': all_epoch_accs
    }

def create_reconstruction_viewer(save_dir, visualize_every, num_epochs):
    """Create a simple HTML file to view reconstructions across epochs"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vision Transformer Reconstructions Viewer</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .controls { margin-bottom: 20px; }
            .image-container { text-align: center; }
            button { padding: 8px 16px; margin: 0 5px; }
            select { padding: 8px; }
        </style>
    </head>
    <body>
        <h1>Vision Transformer Reconstructions Viewer</h1>
        <div class="controls">
            <button id="prev">Previous</button>
            <select id="epochSelect"></select>
            <button id="next">Next</button>
        </div>
        <div class="image-container">
            <img id="reconstructionImage" style="max-width: 100%;">
        </div>
        
        <script>
            // List of available epochs
            const epochs = [];
            const visualizeEvery = VISUALIZE_EVERY;
            const numEpochs = NUM_EPOCHS;
            
            // Populate epochs array
            for (let i = 0; i <= numEpochs; i += visualizeEvery) {
                epochs.push(i + 1);
            }
            if (epochs[epochs.length - 1] !== numEpochs) {
                epochs.push(numEpochs);
            }
            
            // Populate select element
            const select = document.getElementById('epochSelect');
            for (const epoch of epochs) {
                const option = document.createElement('option');
                option.value = epoch;
                option.textContent = `Epoch ${epoch}`;
                select.appendChild(option);
            }
            
            // Set initial image
            let currentIndex = 0;
            updateImage();
            
            // Event listeners
            document.getElementById('prev').addEventListener('click', () => {
                if (currentIndex > 0) {
                    currentIndex--;
                    select.selectedIndex = currentIndex;
                    updateImage();
                }
            });
            
            document.getElementById('next').addEventListener('click', () => {
                if (currentIndex < epochs.length - 1) {
                    currentIndex++;
                    select.selectedIndex = currentIndex;
                    updateImage();
                }
            });
            
            select.addEventListener('change', () => {
                currentIndex = select.selectedIndex;
                updateImage();
            });
            
            function updateImage() {
                const epoch = epochs[currentIndex];
                document.getElementById('reconstructionImage').src = `reconstructions/epoch_${epoch}.png`;
            }
        </script>
    </body>
    </html>
    """.replace('VISUALIZE_EVERY', str(visualize_every)).replace('NUM_EPOCHS', str(num_epochs))
    
    with open(f"{save_dir}/view_reconstructions.html", 'w') as f:
        f.write(html_content)
    
    print(f"Created reconstruction viewer at {save_dir}/view_reconstructions.html")

# Define a custom WandB callback to configure what to log
def setup_wandb_logging(project_name="vit-discrete-tokens", run_name=None, config=None, log_model=True):
    """Setup WandB with custom configuration to focus on the most important metrics"""
    if run_name is None:
        run_name = f"vit_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize WandB with focused logging
    run = wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        settings=wandb.Settings(
            # Customize what gets logged to WandB
            log_model=log_model,
            silent="warning"  # Reduce WandB logging noise
        )
    )
    
    # Define what metrics to log in WandB
    wandb.define_metric("batch_loss", summary="min")
    wandb.define_metric("batch_accuracy", summary="max")
    wandb.define_metric("epoch_loss", summary="min")
    wandb.define_metric("epoch_accuracy", summary="max")
    wandb.define_metric("cycle_avg_loss", summary="min")
    wandb.define_metric("cycle_avg_accuracy", summary="max")
    
    return run

# Main pipeline with improvements
def run_vit_training_pipeline(
    data_array,
    patch_size=3,
    hidden_dim=128,
    num_heads=8,
    num_layers=6,
    ffn_dim=512,
    dropout=0.1,
    batch_size=32,
    num_epochs=100,
    learning_rate=1e-3,
    min_mask_ratio=0.05,
    max_mask_ratio=0.5,
    cycle_length=10,
    use_wandb=True,
    wandb_project="vit-discrete-tokens",
    wandb_name=None,
    save_dir='./model_checkpoints',
    test_split=0.1,  # Proportion of data to use for testing
    visualize_every=5,
    seed=42
):
    """
    Complete pipeline for training Vision Transformer with discrete tokens - improved monitoring
    
    Args:
        data_array: Numpy array or Torch tensor of shape (N, H, W)
        patch_size: Size of image patches
        hidden_dim: Hidden dimension of transformer
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        ffn_dim: Feedforward network dimension
        dropout: Dropout rate
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate
        min_mask_ratio: Minimum masking ratio in cycle
        max_mask_ratio: Maximum masking ratio in cycle
        cycle_length: Length of masking cycle in epochs
        use_wandb: Whether to use Weights & Biases
        wandb_project: W&B project name
        wandb_name: W&B run name
        save_dir: Directory to save model checkpoints
        test_split: Proportion of data to use for testing
        visualize_every: How often to create visualizations (epochs)
        seed: Random seed
    """
    # Set random seed for reproducibility
    set_seed(seed)
    
    # Initialize W&B with focused logging
    if use_wandb:
        config = {
            "patch_size": patch_size,
            "hidden_dim": hidden_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "min_mask_ratio": min_mask_ratio,
            "max_mask_ratio": max_mask_ratio,
            "cycle_length": cycle_length,
            "architecture": "VisionTransformer_DiscreteTokens"
        }
        setup_wandb_logging(wandb_project, wandb_name, config)
    
    # Create output directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/reconstructions", exist_ok=True)
    
    # Process the input array
    print("Processing input data...")
    
    # Convert to numpy if it's a torch tensor
    if isinstance(data_array, torch.Tensor):
        data_array = data_array.numpy()
    
    # Ensure the data is in the right shape (N, H, W)
    if len(data_array.shape) != 3:
        raise ValueError(f"Expected data shape (N, H, W), got {data_array.shape}")
    
    # Split into training and test sets
    n_samples = data_array.shape[0]
    n_test = int(n_samples * test_split)
    
    # Shuffle indices
    indices = np.random.permutation(n_samples)
    train_indices = indices[n_test:]
    test_indices = indices[:n_test]
    
    training_data = data_array[train_indices]
    test_data = data_array[test_indices]
    
    print(f"Training data shape: {training_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    # Convert to torch tensors
    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, dtype=torch.float32)
    
    if not isinstance(test_data, torch.Tensor):
        test_data = torch.tensor(test_data, dtype=torch.float32)
    
    # Normalize data to 0-1 if not already
    if training_data.max() > 1.0:
        training_data = training_data / 255.0
    
    if test_data.max() > 1.0:
        test_data = test_data / 255.0
    
    # Ensure data is in valid range
    assert 0.0 <= training_data.min() and training_data.max() <= 1.0, "Training data should be in range [0, 1]"
    assert 0.0 <= test_data.min() and test_data.max() <= 1.0, "Test data should be in range [0, 1]"
    
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params = build_vocabulary(
        training_data, patch_size, partial_masked_token=True
    )
    print(f"Vocabulary size: {vocab.shape[0]}")
    
    # Initialize model
    print("Initializing model...")
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
        max_rel_dist=32,
        dropout=dropout
    )
    
    # Move model to device
    model = model.to(device)
    
    # Setup optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Train the model with improved monitoring
    print("Starting training...")
    trained_model, metrics = train_vit_with_cyclical_masking(
        model=model,
        training_data=training_data,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=batch_size,
        num_epochs=num_epochs,
        min_mask_ratio=min_mask_ratio,
        max_mask_ratio=max_mask_ratio,
        cycle_length=cycle_length,
        use_wandb=use_wandb,
        save_dir=save_dir,
        visualize_every=visualize_every
    )
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_accuracy = evaluate_model(
        model=trained_model,
        test_data=test_data,
        mask_ratio=0.3,
        patch_size=patch_size,
        device=device
    )
    
    print(f"Final test accuracy: {test_accuracy:.4f}")
    
    if use_wandb:
        wandb.log({"final_test_accuracy": test_accuracy})
        wandb.finish()
    
    print(f"Training complete! Reconstructions viewer available at: {save_dir}/view_reconstructions.html")
    
    return trained_model, metrics, test_accuracy

def evaluate_model(model, test_data, mask_ratio=0.3, patch_size=None, device=None):
    """Evaluate model on test data"""
    if device is None:
        device = next(model.parameters()).device
    
    if patch_size is None:
        patch_size = model.patch_size
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(), tqdm(total=len(test_data), desc="Evaluating") as eval_pbar:
        for i in range(len(test_data)):
            sample = test_data[i].to(device)
            
            # Create a masked version
            masked_sample = sample.clone()
            H, W = sample.shape
            n_h = H // patch_size
            n_w = W // patch_size
            total_patches = n_h * n_w
            
            # Choose random patches to mask
            mask_indices = np.random.choice(
                total_patches, 
                int(mask_ratio * total_patches), 
                replace=False
            )
            
            # Apply masking
            for idx in mask_indices:
                i_patch = idx // n_w
                j_patch = idx % n_w
                masked_sample[i_patch*patch_size:(i_patch+1)*patch_size, 
                              j_patch*patch_size:(j_patch+1)*patch_size] = 0.5
            
            # Extract patches
            patches = []
            for pi in range(n_h):
                for pj in range(n_w):
                    original_patch = sample[pi*patch_size:(pi+1)*patch_size, 
                                           pj*patch_size:(pj+1)*patch_size].reshape(-1)
                    masked_patch = masked_sample[pi*patch_size:(pi+1)*patch_size, 
                                                pj*patch_size:(pj+1)*patch_size].reshape(-1)
                    
                    patches.append(masked_patch)
                    
                    # Find target token
                    target_idx = model._find_closest_token(original_patch)
                    
                    # Check if this patch was masked
                    was_masked = torch.any(torch.abs(masked_patch - 0.5) < 0.1)
                    if was_masked:
                        # Add to test set
                        total += 1
            
            # Process all patches at once
            patches_tensor = torch.stack(patches).unsqueeze(0).to(device)
            
            # Forward pass
            logits, _ = model(patches_tensor)
            
            # Get predictions
            predictions = torch.argmax(logits[0], dim=1)
            
            # Compare predictions with targets, only for masked patches
            for idx in mask_indices:
                target_patch = sample[idx//n_w*patch_size:(idx//n_w+1)*patch_size, 
                                     idx%n_w*patch_size:(idx%n_w+1)*patch_size].reshape(-1)
                target_idx = model._find_closest_token(target_patch)
                
                if predictions[idx] == target_idx:
                    correct += 1
            
            eval_pbar.update(1)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


