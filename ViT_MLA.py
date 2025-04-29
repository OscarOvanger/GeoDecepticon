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
    if not isinstance(training_data, torch.Tensor):
        training_data = torch.tensor(training_data, dtype=torch.float32)
    N, H, W = training_data.shape
    patches = training_data.unfold(1, patch_size, patch_size)
    patches = patches.unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(N, -1, patch_size * patch_size)
    patches_flat = patches.view(-1, patch_size * patch_size)
    vocab, counts = torch.unique(patches_flat, dim=0, return_counts=True)
    sorted_idx = torch.argsort(counts, descending=True)
    vocab = vocab[sorted_idx][:cap_size]
    counts = counts[sorted_idx][:cap_size]
    mask_token = nn.Parameter(torch.full((patch_size * patch_size,), 0.5), requires_grad=True)
    return vocab, counts, mask_token

########################################
# Helper: Convert Patches to Image
########################################

def patches_to_image(patches, img_shape, patch_size):
    H, W = img_shape
    ph = H // patch_size
    pw = W // patch_size
    img = patches.view(ph, pw, patch_size, patch_size)
    img = img.permute(0,2,1,3).contiguous().view(H, W)
    return img

########################################
# Multi-Head Latent Attention Block
########################################


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RopelessMLAEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, latent_dim, ffn_dim=None):
        super().__init__()
        assert emb_dim % num_heads == 0
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.latent_dim = latent_dim

        # 1) Q, Wa_K, Wa_V, Wb_K, Wb_V
        self.Wq      = nn.Linear(emb_dim, emb_dim)
        self.Wa_K    = nn.Linear(emb_dim, latent_dim)
        self.Wb_K    = nn.Linear(latent_dim, emb_dim)
        self.Wa_V    = nn.Linear(emb_dim, latent_dim)
        self.Wb_V    = nn.Linear(latent_dim, emb_dim)

        # 2) output proj & FFN
        self.proj_o  = nn.Linear(emb_dim, emb_dim)
        ffn_dim = ffn_dim or emb_dim * 4
        self.ffn     = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim)
        )

        # 3) LayerNorms
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_bias=None):
        B, N, _ = x.size()

        # --- 1) linear projections ---
        Q  = self.Wq(x)                      # [B,N,D]
        Kc = self.Wa_K(x)                    # [B,N,r]
        Vc = self.Wa_V(x)                    # [B,N,r]
        K  = self.Wb_K(Kc)                   # [B,N,D]
        V  = self.Wb_V(Vc)                   # [B,N,D]

        # --- 2) reshape into heads ---
        def reshape_heads(z):
            return z.view(B, N, self.num_heads, self.head_dim) \
                    .permute(0,2,1,3)         # [B,H,N,dh]

        Qh = reshape_heads(Q)
        Kh = reshape_heads(K)
        Vh = reshape_heads(V)

        # --- 3) scaled dot-product attention ---
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            # attn_bias: [N,N]
            scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        AhV = torch.matmul(weights, Vh)      # [B,H,N,dh]

        # --- 4) concat heads and project out ---
        O = AhV.permute(0,2,1,3).contiguous().view(B, N, self.emb_dim)
        O = self.proj_o(O)
        x = self.norm1(x + O)

        # --- 5) feed-forward + residual ---
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x

class StackedContextViT_MLA(nn.Module):
    def __init__(self,
                 vocab,                # Tensor [V,D]
                 mask_token,           # Parameter [D]
                 patch_dim,            # D
                 num_patches,          # N
                 emb_dim=128,
                 num_heads=4,
                 latent_dim=None,
                 num_layers=4,
                 ffn_dim=None
                ):
        super().__init__()
        self.vocab       = vocab
        self.vocab_size  = vocab.size(0)
        self.mask_token  = mask_token
        self.patch_proj  = nn.Linear(patch_dim, emb_dim)
        self.pos_emb     = nn.Parameter(torch.zeros(num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # relative bias (optional — you can also use absolute pos)
        self.rel_bias    = nn.Parameter(torch.zeros(num_patches, num_patches))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)

        latent_dim = latent_dim or emb_dim
        self.layers = nn.ModuleList([
            RopelessMLAEncoderBlock(emb_dim, num_heads, latent_dim, ffn_dim)
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, patches, mask_rate=0.):
        """
        patches: [B,N,patch_dim], values in [0,1]
        mask_rate: fraction of patches to mask each forward
        """
        B, N, _ = patches.size()
        mask = torch.rand(B, N, device=patches.device) < mask_rate

        # 1) replace masked with mask token
        x = patches.clone()
        x[mask] = self.mask_token.to(x.device)

        # 2) embed + pos
        x = self.patch_proj(x) + self.pos_emb.unsqueeze(0)

        # 3) MLA blocks
        bias = self.rel_bias[:N, :N]
        for block in self.layers:
            x = block(x, attn_bias=bias)

        # 4) final logits
        logits = self.out_proj(x)   # [B,N,V]
        return logits, mask

########################################
# Dataset Class
########################################

class BinaryImageDataset(Dataset):
    def __init__(self, images):
        self.images = images
    def __len__(self):
        return self.images.size(0)
    def __getitem__(self, idx):
        return self.images[idx]
    @staticmethod
    def batch_to_patches(images, patch_size):
        B, H, W = images.size()
        patches = images.unfold(1, patch_size, patch_size)
        patches = patches.unfold(2, patch_size, patch_size)
        return patches.contiguous().view(B, -1, patch_size * patch_size)
