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
    def __init__(self, emb_dim, num_heads, ffn_dim,dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim,
                                          num_heads=num_heads,
                                          dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.ffn   = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, emb_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_mask=None):
        # attn_mask here is [N,N] if provided
        out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + self.drop1(out))
        x = self.norm2(x + self.ffn(x))
        return x

class StackedContextViT(nn.Module):
    def __init__(
        self,
        vocab,                # [V, patch_dim]
        mask_token,           # Parameter [patch_dim]
        patch_dim,
        num_patches,          # N
        emb_dim=128,
        num_heads=1,
        num_layers=1,
        ffn_dim=None,
        use_pos_emb: bool = True,
        pos_emb_init: torch.Tensor | None = None,
        use_rel_bias: bool = True,
        rel_bias_init: torch.Tensor | None = None,  # if provided must be [2G-1,2G-1]
    ):
        super().__init__()
        self.vocab      = vocab
        self.mask_token = mask_token
        self.vocab_size = vocab.size(0)
        self.patch_proj = nn.Linear(patch_dim, emb_dim)
        self.out_drop = nn.Dropout(0.1)

        # absolute positional embedding
        self.use_pos_emb = use_pos_emb
        if use_pos_emb:
            pe = pos_emb_init if (pos_emb_init is not None) \
                 else torch.zeros(num_patches, emb_dim)
            self.pos_emb = nn.Parameter(pe)
        else:
            # dummy so attribute exists, but never used
            self.register_buffer('pos_emb', torch.zeros(1))

        # stationary relative‐bias table
        self.use_rel_bias = use_rel_bias
        G = int(math.sqrt(num_patches))
        if use_rel_bias:
            # table of size (2G-1)×(2G-1)
            if rel_bias_init is not None:
                assert rel_bias_init.shape == (2*G-1, 2*G-1)
                rb = rel_bias_init
            else:
                rb = torch.zeros(2*G-1, 2*G-1)
            self.rel_bias_table = nn.Parameter(rb)
            self.G = G
        else:
            self.register_buffer('rel_bias_table', torch.zeros(1))
            self.G = None

        # transformer layers
        ffn_dim = ffn_dim or (emb_dim * 4)
        self.layers = nn.ModuleList([
            TransformerEncoderBlock(emb_dim, num_heads, ffn_dim)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, patches, mask_rate):
        """
        patches: [B, N, patch_dim]
        """
        B, N, _ = patches.shape
        G = self.G  # sqrt(N) if rel_bias

        # 1) mask
        mask = torch.rand(B, N, device=patches.device) < mask_rate
        x = patches.clone()
        x[mask] = self.mask_token.to(x.device)

        # 2) patch → embedding
        x = self.patch_proj(x)

        # 3) add absolute pos‐emb if desired
        if self.use_pos_emb:
            x = x + self.pos_emb.unsqueeze(0)

        # 4) build the [N×N] attention mask from stationary table
        attn_bias = None
        if self.use_rel_bias:
            # compute row/col indices for each patch 0..N-1
            idx = torch.arange(N, device=x.device)
            rows = idx // G   # shape [N]
            cols = idx %  G   # shape [N]

            # pairwise offsets
            dr = rows[:,None] - rows[None,:]   # [N,N] in range [-(G-1),(G-1)]
            dc = cols[:,None] - cols[None,:]

            # shift to [0..2G-2]
            ir = dr + (G-1)
            ic = dc + (G-1)

            attn_bias = self.rel_bias_table[ir, ic]  # [N,N]

        # 5) apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_bias)

        # 6) final logits
        logits = self.out_drop(self.out_proj(x))  # [B, N, V]
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


