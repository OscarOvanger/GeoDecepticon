import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import wandb
from tqdm.auto import tqdm
import random

# Core utilities
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def training_dat_to_patches(training_data, patch_size):
    N, H, W = training_data.shape
    n_h, n_w = H // patch_size, W // patch_size
    x = training_data.reshape(N, 1, H, W)
    patches = F.unfold(x, kernel_size=patch_size, stride=patch_size)
    return patches.transpose(1, 2).reshape(-1, patch_size * patch_size)

def build_vocabulary(training_data, patch_size, partial_masked_token=True):
    patches = training_dat_to_patches(training_data, patch_size)
    
    # Create binary patterns
    vocab = [torch.tensor([int(bit) for bit in np.binary_repr(i, width=patch_size**2)], 
                          dtype=torch.float32) for i in range(2**(patch_size**2))]
    
    # Add masked token
    fully_masked_token = torch.ones(patch_size**2) * 0.5
    vocab.append(fully_masked_token)
    fully_masked_idx = len(vocab) - 1
    
    # Add partial masked tokens
    partial_token_indices = []
    partial_masked_token_params = []
    
    if partial_masked_token:
        for pos in range(patch_size**2):
            for val in range(2):
                token = torch.ones(patch_size**2) * 0.5
                token[pos] = float(val)
                vocab.append(token)
                partial_token_indices.append(len(vocab) - 1)
                
                # Trainable token
                mask = patches[:, pos] == val
                if mask.any():
                    param = nn.Parameter(torch.mean(patches[mask].float(), dim=0).clone(), requires_grad=True)
                else:
                    param = nn.Parameter(fully_masked_token.clone(), requires_grad=True)
                    param[pos] = float(val)
                
                partial_masked_token_params.append(param)
    
    # Create vocabulary tensor and masked token param
    vocab = torch.stack(vocab)
    masked_token_param = nn.Parameter(torch.mean(patches.float(), dim=0), requires_grad=True)
    
    return vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params

# Model components
class RelativePositionSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, max_rel_dist=32, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.max_rel_dist = max_rel_dist
        self.rel_pos_bias = nn.Parameter(torch.zeros(2 * max_rel_dist + 1, num_heads))
        nn.init.normal_(self.rel_pos_bias, std=0.02)
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Generate relative position bias
        i = torch.arange(seq_len).unsqueeze(1)
        j = torch.arange(seq_len).unsqueeze(0)
        rel_pos = torch.clamp(i - j, -self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
        rel_pos = rel_pos.to(x.device)
        
        rel_pos_bias = self.rel_pos_bias[rel_pos].permute(2, 0, 1)
        
        # Apply attention with mask
        attn_mask = torch.zeros(seq_len, seq_len, device=x.device)
        for h in range(rel_pos_bias.size(0)):
            attn_mask += rel_pos_bias[h]
        
        attn_mask = attn_mask / rel_pos_bias.size(0)
        output, _ = self.mha(x, x, x, attn_mask=attn_mask)
        
        return output

class TransformerEncoderLayer(nn.Module):
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
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm)
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, vocab, fully_masked_idx, partial_token_indices, 
                 masked_token_param, partial_masked_token_params, patch_size, 
                 num_heads, num_layers, ffn_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.register_buffer('vocab', vocab)
        self.fully_masked_idx = fully_masked_idx
        self.partial_token_indices = partial_token_indices
        self.masked_token_param = masked_token_param
        self.partial_masked_token_params = nn.ParameterList(partial_masked_token_params)
        
        self.patch_size = patch_size
        self.vocab_size = vocab.shape[0]
        
        self.token_embedding = nn.Linear(patch_size**2, hidden_dim)
        self.transformer = nn.Sequential(*[
            TransformerEncoderLayer(hidden_dim, num_heads, ffn_dim, dropout, max_rel_dist=32)
            for _ in range(num_layers)
        ])
        self.output_projection = nn.Linear(hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.zeros_(self.token_embedding.bias)
        nn.init.normal_(self.output_projection.weight, std=0.02)
        nn.init.zeros_(self.output_projection.bias)
    
    def _find_closest_token(self, patch):
        distances = torch.sum((self.vocab - patch.unsqueeze(0))**2, dim=1)
        return torch.argmin(distances)
    
    def _process_batch(self, patches):
        B, num_patches, patch_dim = patches.shape
        device = patches.device
        
        is_fully_masked = torch.all(torch.abs(patches - 0.5) < 0.1, dim=-1)
        processed_patches = torch.zeros_like(patches)
        token_indices = torch.zeros((B, num_patches), dtype=torch.long, device=device)
        
        for b in range(B):
            for i in range(num_patches):
                patch = patches[b, i]
                
                if is_fully_masked[b, i]:
                    token_indices[b, i] = self.fully_masked_idx
                    processed_patches[b, i] = self.masked_token_param
                else:
                    mask_positions = torch.abs(patch - 0.5) < 0.1
                    binary_positions = ~mask_positions
                    
                    if torch.any(mask_positions) and torch.any(binary_positions):
                        for pos in range(patch_dim):
                            if binary_positions[pos]:
                                val = round(patch[pos].item())
                                partial_idx = pos * 2 + val
                                if partial_idx < len(self.partial_token_indices):
                                    token_indices[b, i] = self.partial_token_indices[partial_idx]
                                    processed_patches[b, i] = self.partial_masked_token_params[partial_idx]
                                break
                    else:
                        idx = self._find_closest_token(patch)
                        token_indices[b, i] = idx
                        processed_patches[b, i] = patch
        
        return processed_patches, token_indices
    
    def forward(self, patches):
        processed_patches, token_indices = self._process_batch(patches)
        embeddings = self.token_embedding(processed_patches)
        embeddings = self.dropout(embeddings)
        z = self.transformer(embeddings)
        logits = self.output_projection(z)
        return logits, token_indices

# Training functions
def test_reconstruction(model, original_sample, masked_sample):
    H, W = original_sample.shape
    patch_size = model.patch_size
    n_h, n_w = H // patch_size, W // patch_size
    
    # Extract patches
    patches = []
    mask_indices = []
    
    for i in range(n_h):
        for j in range(n_w):
            patch = masked_sample[i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size]
            patches.append(patch.reshape(-1))
            
            # Check if masked
            if torch.any(torch.abs(patch - 0.5) < 0.1):
                mask_indices.append(len(patches) - 1)
    
    patches_tensor = torch.stack(patches).unsqueeze(0).to(masked_sample.device)
    
    # Forward pass
    with torch.no_grad():
        logits, _ = model(patches_tensor)
    
    # Create reconstruction
    token_indices = torch.argmax(logits, dim=-1)[0]
    reconstructed = original_sample.clone()
    
    for idx in mask_indices:
        i = (idx // n_w) * patch_size
        j = (idx % n_w) * patch_size
        token = model.vocab[token_indices[idx]].reshape(patch_size, patch_size)
        reconstructed[i:i+patch_size, j:j+patch_size] = token
    
    return reconstructed

def calculate_accuracy(model, batch_patches, masked_patches, mask_indices, targets):
    with torch.no_grad():
        logits, _ = model(masked_patches)
    
    predictions = torch.argmax(logits, dim=-1)
    batch_size = batch_patches.shape[0]
    
    correct = 0
    total = 0
    
    for b in range(batch_size):
        for idx in mask_indices[b]:
            correct += (predictions[b, idx] == targets[b, idx]).item()
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def get_mask_ratio(epoch, min_ratio=0.05, max_ratio=0.5, cycle_length=10):
    cycle = epoch // cycle_length
    epoch_in_cycle = epoch % cycle_length
    ratio = min_ratio + (max_ratio - min_ratio) * (epoch_in_cycle / (cycle_length - 1))
    return ratio, cycle, epoch_in_cycle

def train_vit(model, train_data, test_data, batch_size=32, num_epochs=100, 
             min_mask_ratio=0.05, max_mask_ratio=0.5, cycle_length=10, 
             use_wandb=True, save_dir='./checkpoints'):
    device = next(model.parameters()).device
    os.makedirs(save_dir, exist_ok=True)
    
    # Select test samples for quick evaluation
    test_indices = np.random.choice(len(test_data), min(3, len(test_data)), replace=False)
    test_samples = test_data[test_indices].to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training metrics
    cycle_losses = []
    cycle_accs = []
    
    # Training loop
    for epoch in range(num_epochs):
        mask_ratio, cycle, epoch_in_cycle = get_mask_ratio(
            epoch, min_mask_ratio, max_mask_ratio, cycle_length
        )
        
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        
        # Get image dimensions
        H, W = train_data.shape[1:]
        n_h, n_w = H // model.patch_size, W // model.patch_size
        patches_per_image = n_h * n_w
        num_batches = (len(train_data) + batch_size - 1) // batch_size
        
        # Process in batches
        with tqdm(total=num_batches, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx in range(num_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(train_data))
                batch_size_actual = end_idx - start_idx
                batch_images = train_data[start_idx:end_idx].to(device)
                
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
                
                # Create masked version
                masked_patches = batch_patches.clone()
                mask_indices = []
                
                for b in range(batch_size_actual):
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
                
                # Compute targets
                targets = torch.zeros(batch_size_actual, patches_per_image, dtype=torch.long, device=device)
                for b in range(batch_size_actual):
                    for p in range(patches_per_image):
                        targets[b, p] = model._find_closest_token(batch_patches[b, p])
                
                # Forward pass
                optimizer.zero_grad()
                logits, _ = model(masked_patches)
                loss = criterion(logits.reshape(-1, model.vocab_size), targets.reshape(-1))
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Calculate accuracy
                batch_acc, correct, total = calculate_accuracy(
                    model, batch_patches, masked_patches, mask_indices, targets
                )
                
                # Update metrics
                epoch_loss += loss.item() * batch_size_actual
                epoch_correct += correct
                epoch_total += total
                
                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{batch_acc:.4f}"})
                pbar.update(1)
                
                # Log to W&B
                if use_wandb and batch_idx % 10 == 0:
                    wandb.log({
                        'batch_loss': loss.item(),
                        'batch_accuracy': batch_acc,
                        'mask_ratio': mask_ratio,
                    })
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_data)
        avg_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        # Store metrics
        cycle_losses.append(avg_loss)
        cycle_accs.append(avg_acc)
        
        # Log epoch metrics
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, Mask Ratio: {mask_ratio:.2f}")
        
        if use_wandb:
            wandb.log({
                'epoch': epoch,
                'epoch_loss': avg_loss,
                'epoch_accuracy': avg_acc,
            })
        
        # End of cycle
        if (epoch + 1) % cycle_length == 0 or (epoch + 1) == num_epochs:
            cycle_avg_loss = np.mean(cycle_losses)
            cycle_avg_acc = np.mean(cycle_accs)
            
            print(f"Cycle {cycle+1} completed, Avg Loss: {cycle_avg_loss:.4f}, Avg Acc: {cycle_avg_acc:.4f}")
            
            if use_wandb:
                wandb.log({
                    'cycle_avg_loss': cycle_avg_loss,
                    'cycle_avg_accuracy': cycle_avg_acc,
                })
            
            # Save model checkpoint
            torch.save(model.state_dict(), f"{save_dir}/model_cycle_{cycle+1}.pt")
            
            # Evaluate on test samples
            if (epoch + 1) % cycle_length == 0:
                model.eval()
                with torch.no_grad():
                    # Show reconstruction of a few samples
                    for i, sample in enumerate(test_samples):
                        # Create masked version
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
                        
                        # Plot if last cycle
                        if (epoch + 1) == num_epochs:
                            plt.figure(figsize=(12, 4))
                            plt.subplot(1, 3, 1)
                            plt.imshow(sample.cpu().numpy(), cmap='gray')
                            plt.title("Original")
                            plt.axis('off')
                            
                            plt.subplot(1, 3, 2)
                            plt.imshow(masked_sample.cpu().numpy(), cmap='gray')
                            plt.title("Masked")
                            plt.axis('off')
                            
                            plt.subplot(1, 3, 3)
                            plt.imshow(reconstructed.cpu().numpy(), cmap='gray')
                            plt.title("Reconstructed")
                            plt.axis('off')
                            
                            plt.tight_layout()
                            plt.savefig(f"{save_dir}/test_sample_{i}_cycle_{cycle+1}.png")
                            plt.close()
            
            # Reset cycle metrics
            cycle_losses = []
            cycle_accs = []
    
    # Save final model
    torch.save(model.state_dict(), f"{save_dir}/model_final.pt")
    
    return model

def run_training(data_array, patch_size=3, hidden_dim=128, num_heads=8, num_layers=6, 
                ffn_dim=512, dropout=0.1, batch_size=32, num_epochs=100, 
                min_mask_ratio=0.05, max_mask_ratio=0.5, cycle_length=10, 
                use_wandb=True, save_dir='./checkpoints', test_split=0.1, seed=42):
    # Setup
    set_seed(seed)
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
    
    # Build vocabulary and model
    vocab, fully_masked_idx, partial_token_indices, masked_token_param, partial_masked_token_params = build_vocabulary(
        train_data, patch_size)
    
    # Create model
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
    
    # Initialize wandb if needed
    if use_wandb:
        wandb.init(
            project="vit-discrete-tokens",
            config={
                "patch_size": patch_size,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "ffn_dim": ffn_dim
            }
        )
    
    # Train model
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


