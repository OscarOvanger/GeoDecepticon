import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # For progress bars
import matplotlib.patches as mpatches  # For drawing colored borders
import math

########################################
# Build Vocabulary and Mask Token
########################################

def build_vocabulary(training_data, patch_size, cap_size):
    """
    Scans all non-overlapping patches in training_data and builds a vocabulary of unique patches.

    Args:
        training_data (torch.Tensor or numpy.array): Tensor of shape (N, H, W).
        patch_size (int): Size of the patch (e.g., 8 for an 8x8 patch).

    Returns:
        vocab (torch.Tensor): Unique patches, shape (num_unique, patch_size*patch_size).
        counts (torch.Tensor): Count for each unique patch, shape (num_unique,).
        mask_token (torch.Tensor): Mask token, a tensor of shape (patch_size*patch_size,) with all values 0.5.
    """
    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, dtype=torch.float32)

    N, H, W = training_data.shape

    # Extract non-overlapping patches using unfold.
    patches = training_data.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(N, -1, patch_size * patch_size)
    patches_flat = patches.view(-1, patch_size * patch_size)

    # Find unique patches and their counts.
    vocab, counts = torch.unique(patches_flat, dim=0, return_counts=True)
    sorted_indices = torch.argsort(counts, descending=True)
    vocab = vocab[sorted_indices][:cap_size]
    counts = counts[sorted_indices][:cap_size]

    # Create a learnable mask token, initialized to 0.5.
    mask_token_torch = torch.full((patch_size * patch_size,), 0.5, dtype=torch.float32)
    mask_token = nn.Parameter(mask_token_torch, requires_grad=True)

    return vocab, counts, mask_token

########################################
# Helper: Convert Patches to Image
########################################

def patches_to_image(patches, img_shape, patch_size):
    """
    Reconstructs an image from patches.

    Args:
        patches (torch.Tensor): Tensor of shape [num_patches, patch_dim].
        img_shape (tuple): (H, W) of the original image.
        patch_size (int): The side length of a square patch.

    Returns:
        image (torch.Tensor): Reconstructed image of shape (H, W).
    """
    H, W = img_shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    image = patches.view(num_patches_h, num_patches_w, patch_size, patch_size)
    image = image.permute(0, 2, 1, 3).contiguous().view(H, W)
    return image

########################################
# Transformer Encoder Block and Stacked ViT
########################################

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, ffn_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class StackedContextViT(nn.Module):
    def __init__(
        self,
        vocab,                # Tensor [V, D]
        mask_token,           # Parameter [D]
        patch_dim,            # D
        num_patches,          # N
        emb_dim=128,
        num_heads=1,
        num_layers=1,
        ffn_dim=None,

        # new args:
        use_pos_emb: bool = True,
        pos_emb_init: torch.Tensor | None = None,   # if provided, must be [N, emb_dim]
        use_rel_bias: bool = True,
        rel_bias_init: torch.Tensor | None = None   # if provided, must be [N, N]
    ):
        super().__init__()
        self.vocab      = vocab
        self.vocab_size = vocab.size(0)
        self.patch_dim  = patch_dim
        self.emb_dim    = emb_dim

        # patch embedding
        self.patch_proj = nn.Linear(patch_dim, emb_dim)

        # positional encoding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            if pos_emb_init is not None:
                # user supplied values
                assert pos_emb_init.shape == (num_patches, emb_dim)
                pe = pos_emb_init
            else:
                # zero init
                pe = torch.zeros(num_patches, emb_dim)
            # always wrap in Parameter so .to(device) will move it
            self.pos_emb = nn.Parameter(pe)
        else:
            # no parameter at all
            self.register_buffer('pos_emb', torch.zeros(1))  # dummy so attribute exists
            # but will ignore it in forward

        # relative bias
        self.use_rel_bias = use_rel_bias
        if use_rel_bias:
            if rel_bias_init is not None:
                assert rel_bias_init.shape == (num_patches, num_patches)
                rb = rel_bias_init
            else:
                rb = torch.zeros(num_patches, num_patches)
            self.rel_bias = nn.Parameter(rb)
        else:
            self.register_buffer('rel_bias', torch.zeros(1))

        # encoder layers
        ffn_dim = ffn_dim or emb_dim * 4
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        # final projection
        self.out_proj = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, patches, mask_rate):
        B, N, P = patches.shape

        # 1) mask
        mask = torch.rand(B, N, device=patches.device) < mask_rate
        x = patches.clone()
        x[mask] = self.mask_token.to(x.device)

        # 2) patch→emb
        x = self.patch_proj(x)

        # 3) add pos emb if enabled
        if self.use_pos_emb:
            # pos_emb is [N, emb_dim]
            x = x + self.pos_emb.unsqueeze(0)

        # 4) build attention bias
        attn_bias = None
        if self.use_rel_bias:
            # rel_bias is [N, N]
            attn_bias = self.rel_bias[:N, :N]

        # 5) transformer
        for layer in self.encoder_layers:
            x = layer(x, attn_mask=attn_bias)

        # 6) to logits
        logits = self.out_proj(x)
        return logits, mask

########################################
# Dataset Class and Utility
########################################

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        """
        images: Tensor of shape [N, H, W] (binary images)
        """
        self.images = images

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]

    @staticmethod
    def batch_to_patches(images, patch_size):
        B, H, W = images.shape
        patches = images.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
        patches = patches.contiguous().view(B, -1, patch_size * patch_size)
        return patches


