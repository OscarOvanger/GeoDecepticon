import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict

class VisionTransformer(nn.Module):
    def __init__(self, num_heads, feedforward_dim, num_layers, d_model, 
                 max_patches=256, dropout=0.0, num_partial_masks=32):
        super().__init__()
        
        # Store configuration
        self.d_model = d_model
        self.max_patches = max_patches
        
        # Build vocabulary dynamically from observed patches
        self.patch_to_token = {}  # Will be populated during training setup
        self.token_to_patch = {}  # Reverse mapping
        self.num_observed_tokens = 0  # Will be set when vocabulary is built
        
        # Special tokens
        self.mask_token_id = None  # Will be set when vocabulary is built
        self.partial_mask_token_ids = []  # Will be populated when vocabulary is built
        
        # Total number of tokens (will be updated when vocabulary is built)
        self.num_tokens = 0
        
        # Token embeddings (will be initialized after vocabulary is built)
        self.token_embedding = None
        
        # Positional embeddings
        self.positional_embedding = nn.Parameter(torch.randn(max_patches, d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer (will be initialized after vocabulary is built)
        self.fc_out = None
    
    def build_vocabulary(self, train_dataset, num_partial_masks=32):
        """
        Build vocabulary from observed 4x4 patches in the training dataset
        
        Args:
            train_dataset: Dataset containing binary images
            num_partial_masks: Number of partial mask tokens to create
        """
        print("Building vocabulary from training data...")
        
        # Count patch occurrences
        patch_counter = defaultdict(int)
        
        # Process all images in the training dataset
        for i in range(len(train_dataset)):
            img = train_dataset[i]
            if isinstance(img, tuple):  # Handle case where dataset returns (image, label)
                img = img[0]
                
            # Convert to numpy if it's a tensor
            if isinstance(img, torch.Tensor):
                img = img.numpy()
                
            # Make sure img is 2D binary
            assert img.ndim == 2 or (img.ndim == 3 and img.shape[0] == 1), "Image must be 2D binary"
            if img.ndim == 3:
                img = img.squeeze(0)
            
            # For each 4x4 patch in the image
            h, w = img.shape
            for y in range(0, h, 4):
                for x in range(0, w, 4):
                    if y+4 <= h and x+4 <= w:  # Ensure patch is within bounds
                        patch = img[y:y+4, x:x+4]
                        # Convert patch to tuple for hashability
                        patch_tuple = tuple(patch.flatten())
                        patch_counter[patch_tuple] += 1
        
        # Sort patches by frequency (most common first)
        sorted_patches = sorted(patch_counter.items(), key=lambda x: x[1], reverse=True)
        
        # Create token mappings
        for i, (patch_tuple, _) in enumerate(sorted_patches):
            self.patch_to_token[patch_tuple] = i
            self.token_to_patch[i] = torch.tensor(patch_tuple, dtype=torch.float32)
            
        self.num_observed_tokens = len(self.patch_to_token)
        print(f"Found {self.num_observed_tokens} unique patches in training data")
        
        # Add mask token
        self.mask_token_id = self.num_observed_tokens
        
        # Add partial mask tokens (masked with only one observed cell)
        self.partial_mask_token_ids = []
        for position in range(16):  # 4x4 = 16 positions
            for value in [0, 1]:  # Binary values
                if len(self.partial_mask_token_ids) < num_partial_masks:
                    self.partial_mask_token_ids.append(self.mask_token_id + 1 + len(self.partial_mask_token_ids))
        
        # Update total number of tokens
        self.num_tokens = self.num_observed_tokens + 1 + len(self.partial_mask_token_ids)
        print(f"Total token vocabulary size: {self.num_tokens}")
        
        # Now initialize the embedding table
        self.token_embedding = nn.Embedding(self.num_tokens, self.d_model)
        
        # Initialize output layer
        self.fc_out = nn.Linear(self.d_model, self.num_tokens)
    
    def tokenize_image(self, img, mask=None):
        """
        Convert a binary image to token indices
        
        Args:
            img: Binary image of shape (H, W) or (1, H, W)
            mask: Optional binary mask of same shape, 1 indicates masked area
            
        Returns:
            tokens: Tensor of token indices of shape (num_patches,)
        """
        # Make sure img is 2D
        if img.ndim == 3:
            img = img.squeeze(0)
            
        # Initialize token sequence
        h, w = img.shape
        num_patches_y = h // 4
        num_patches_x = w // 4
        num_patches = num_patches_y * num_patches_x
        tokens = torch.zeros(num_patches, dtype=torch.long)
        
        # Process each patch
        patch_idx = 0
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                if y+4 <= h and x+4 <= w:  # Ensure patch is within bounds
                    # Extract patch
                    patch = img[y:y+4, x:x+4]
                    
                    # Check if patch should be masked based on mask input
                    is_masked = False
                    if mask is not None:
                        patch_mask = mask[y:y+4, x:x+4]
                        if patch_mask.sum() > 0:  # At least one pixel is masked
                            is_masked = True
                            
                            # Check if it's a partial mask with only one observed cell
                            if patch_mask.sum() == 15:  # 15 out of 16 pixels are masked
                                # Find the observed position
                                observed_pos = ((patch_mask == 0).flatten()).nonzero()[0].item()
                                observed_val = patch.flatten()[observed_pos].item()
                                
                                # Calculate index for partial mask token
                                partial_idx = observed_pos * 2 + int(observed_val)
                                if partial_idx < len(self.partial_mask_token_ids):
                                    tokens[patch_idx] = self.partial_mask_token_ids[partial_idx]
                                else:
                                    tokens[patch_idx] = self.mask_token_id
                            else:
                                tokens[patch_idx] = self.mask_token_id
                    
                    if not is_masked:
                        # Convert patch to tuple for dictionary lookup
                        patch_tuple = tuple(patch.flatten().cpu().numpy())
                        
                        # Get token index or use mask token if not in vocabulary
                        if patch_tuple in self.patch_to_token:
                            tokens[patch_idx] = self.patch_to_token[patch_tuple]
                        else:
                            # Handle unknown patch (find nearest neighbor)
                            min_distance = float('inf')
                            nearest_token = 0
                            
                            for token, patch_tensor in self.token_to_patch.items():
                                distance = torch.sum((torch.tensor(patch_tuple) - patch_tensor)**2).item()
                                if distance < min_distance:
                                    min_distance = distance
                                    nearest_token = token
                                    
                            tokens[patch_idx] = nearest_token
                    
                    patch_idx += 1
        
        return tokens

    def forward(self, patches):
        """
        Forward pass through the model
        
        Args:
            patches: Tensor of token indices (B, P) where B is batch size and P is number of patches
            
        Returns:
            logits: Tensor of logits (B, P, V) where V is vocabulary size
        """
        batch_size, seq_len = patches.shape
        
        # Get token embeddings
        token_emb = self.token_embedding(patches)  # (B, P, D)
        
        # Add positional embeddings
        pos_emb = self.positional_embedding[:seq_len].unsqueeze(0).expand(batch_size, -1, -1)  # (B, P, D)
        x = token_emb + pos_emb  # (B, P, D)
        
        # Prepare for transformer (seq_len, batch_size, d_model)
        x = x.permute(1, 0, 2)  # (P, B, D)
        
        # Pass through transformer layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Back to (batch_size, seq_len, d_model)
        x = x.permute(1, 0, 2)  # (B, P, D)
        
        # Output logits
        logits = self.fc_out(x)  # (B, P, V)
        
        return logits

    def get_probabilities(self, logits):
        """
        Compute probabilities using softmax
        
        Args:
            logits: Tensor of logits (B, P, V)
            
        Returns:
            probs: Tensor of probabilities (B, P, V)
        """
        return torch.softmax(logits, dim=-1)
    
    def sample_patches(self, logits, temperature=1.0):
        """
        Sample patches from logits
        
        Args:
            logits: Tensor of logits (B, P, V)
            temperature: Temperature for sampling (higher = more random)
            
        Returns:
            sampled_tokens: Tensor of sampled token indices (B, P)
        """
        # Apply temperature
        logits = logits / temperature
        
        # Get probabilities
        probs = self.get_probabilities(logits)
        
        # Sample from the distribution
        sampled_tokens = torch.multinomial(probs.view(-1, self.num_tokens), 1).view(logits.shape[0], -1)
        
        return sampled_tokens
    
    def generate_image(self, batch_size=1, size=(64, 64), temperature=1.0, device='cpu'):
        """
        Generate new images using autoregressive sampling
        
        Args:
            batch_size: Number of images to generate
            size: Size of the images (H, W)
            temperature: Temperature for sampling
            device: Device to use
            
        Returns:
            images: Generated images (B, 1, H, W)
        """
        h, w = size
        num_patches_y = h // 4
        num_patches_x = w // 4
        num_patches = num_patches_y * num_patches_x
        
        # Start with all masked patches
        tokens = torch.full((batch_size, num_patches), self.mask_token_id, device=device)
        
        # Generate patches autoregressively
        for i in range(num_patches):
            # Get logits from current partial sequence
            logits = self.forward(tokens)
            
            # Sample next token for the current position
            next_token_logits = logits[:, i, :]
            next_token_probs = torch.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(next_token_probs, 1).squeeze(-1)
            
            # Update tokens
            tokens[:, i] = next_token
        
        # Convert tokens back to image
        images = self.tokens_to_image(tokens, size)
        return images
    
    def tokens_to_image(self, tokens, size=(64, 64)):
        """
        Convert token indices to images
        
        Args:
            tokens: Tensor of token indices (B, P)
            size: Size of the output images (H, W)
            
        Returns:
            images: Generated images (B, 1, H, W)
        """
        batch_size = tokens.shape[0]
        h, w = size
        images = torch.zeros((batch_size, 1, h, w), device=tokens.device)
        
        # Convert each token to a 4x4 patch
        patch_idx = 0
        for y in range(0, h, 4):
            for x in range(0, w, 4):
                if y+4 <= h and x+4 <= w and patch_idx < tokens.shape[1]:
                    for b in range(batch_size):
                        token_id = tokens[b, patch_idx].item()
                        
                        # Skip mask tokens (only relevant during inference)
                        if token_id == self.mask_token_id or token_id in self.partial_mask_token_ids:
                            continue
                            
                        # Get patch from token
                        if token_id in self.token_to_patch:
                            patch = self.token_to_patch[token_id].view(4, 4)
                            images[b, 0, y:y+4, x:x+4] = patch
                    
                    patch_idx += 1
        
        return images
    
    def inpaint_image(self, img, mask, temperature=1.0):
        """
        Inpaint masked regions of an image
        
        Args:
            img: Image to inpaint (1, H, W)
            mask: Binary mask where 1 indicates areas to inpaint (1, H, W)
            temperature: Temperature for sampling
            
        Returns:
            inpainted_img: Inpainted image (1, H, W)
        """
        # Tokenize the image with mask
        tokens = self.tokenize_image(img.squeeze(0), mask.squeeze(0)).unsqueeze(0)
        
        # Find masked positions
        masked_positions = []
        for i in range(tokens.size(1)):
            if tokens[0, i] == self.mask_token_id or tokens[0, i] in self.partial_mask_token_ids:
                masked_positions.append(i)
        
        # Inpaint masked patches in a random order
        masked_positions = torch.tensor(masked_positions)
        masked_positions = masked_positions[torch.randperm(len(masked_positions))]
        
        for pos in masked_positions:
            # Get logits
            logits = self.forward(tokens)
            
            # Sample token for the masked position
            pos_logits = logits[0, pos, :]
            pos_probs = torch.softmax(pos_logits / temperature, dim=-1)
            
            # Avoid sampling special tokens
            pos_probs[self.mask_token_id] = 0
            for token_id in self.partial_mask_token_ids:
                pos_probs[token_id] = 0
                
            # Renormalize
            pos_probs = pos_probs / pos_probs.sum()
            
            # Sample
            new_token = torch.multinomial(pos_probs, 1).item()
            
            # Update tokens
            tokens[0, pos] = new_token
        
        # Convert tokens back to image
        inpainted_img = self.tokens_to_image(tokens, size=img.shape[1:])
        return inpainted_img


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Ensure hidden_dim is divisible by num_heads
        assert self.hidden_dim % self.num_heads == 0, "Hidden dimension must be divisible by the number of heads."

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_heads, dropout=dropout)

        # Feedforward network
        self.feedforward = nn.Sequential(
            nn.Linear(self.hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Linear(feedforward_dim, self.hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, z):
        # Apply LayerNorm
        z_norm = self.norm1(z)

        # Self-attention
        attn_output, _ = self.attention(z_norm, z_norm, z_norm)

        # Residual connection
        z = z + self.dropout(attn_output)

        # Feedforward layer
        z_norm = self.norm2(z)
        feedforward_output = self.feedforward(z_norm)

        # Final residual connection
        z = z + self.dropout(feedforward_output)

        return z


# Example usage
def create_model(train_dataset, d_model=512, num_heads=8, num_layers=6, feedforward_dim=2048):
    """
    Create and initialize a Vision Transformer model
    
    Args:
        train_dataset: Dataset containing binary images for vocabulary building
        d_model: Hidden dimension size
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        feedforward_dim: Dimension of feedforward network
        
    Returns:
        model: Initialized VisionTransformer
    """
    model = VisionTransformer(
        num_heads=num_heads,
        feedforward_dim=feedforward_dim,
        num_layers=num_layers,
        d_model=d_model,
        max_patches=256,  # For 64x64 images with 4x4 patches
        dropout=0.1
    )
    
    # Build vocabulary from training data
    model.build_vocabulary(train_dataset)
    
    return model
