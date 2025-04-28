import torch
import torch.nn as nn
import torch.nn.functional as F
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

class MLAEncoderBlock(nn.Module):
    def __init__(self, emb_dim, num_heads, latent_dim, ffn_dim=None):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must be divisible by num_heads"
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads
        self.latent_dim = latent_dim
        # Query projection
        self.Wq = nn.Linear(emb_dim, emb_dim)
        # Latent key/value projections
        self.Wa_K = nn.Linear(emb_dim, latent_dim)
        self.Wb_K = nn.Linear(latent_dim, emb_dim)
        self.Wa_V = nn.Linear(emb_dim, latent_dim)
        self.Wb_V = nn.Linear(latent_dim, emb_dim)
        # Output projection for attention
        self.out_proj_attn = nn.Linear(emb_dim, emb_dim)
        # Feed-forward network
        ffn_dim = ffn_dim or emb_dim * 4
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, emb_dim)
        )
        # Layer norms
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    def forward(self, x, attn_bias=None):
        # x: [B, N, emb_dim]
        B, N, _ = x.size()
        # 1) Project to Q, latent Kc, latent Vc
        Q = self.Wq(x)                  # [B, N, emb_dim]
        Kc = self.Wa_K(x)               # [B, N, latent_dim]
        Vc = self.Wa_V(x)               # [B, N, latent_dim]
        # 2) Expand latents to full K, V
        K = self.Wb_K(Kc)               # [B, N, emb_dim]
        V = self.Wb_V(Vc)               # [B, N, emb_dim]
        # 3) Reshape for multi-head: [B, num_heads, N, head_dim]
        Qh = Q.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        Kh = K.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        Vh = V.view(B, N, self.num_heads, self.head_dim).permute(0,2,1,3)
        # 4) Scaled-dot attention
        scores = torch.matmul(Qh, Kh.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if attn_bias is not None:
            # attn_bias: [N, N]
            scores = scores + attn_bias.unsqueeze(0).unsqueeze(0)
        weights = F.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, Vh)  # [B, heads, N, head_dim]
        # 5) Concat heads and project
        attn_out = attn_out.permute(0,2,1,3).contiguous().view(B, N, self.emb_dim)
        attn_out = self.out_proj_attn(attn_out)
        # 6) Residual + LayerNorm
        x = self.norm1(x + attn_out)
        # 7) Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

########################################
# Stacked Vision Transformer with MLA
########################################

class StackedContextViT(nn.Module):
    def __init__(
        self,
        vocab,
        mask_token,
        patch_dim,
        num_patches,
        emb_dim=128,
        num_heads=4,
        latent_dim=None,
        num_layers=4,
        ffn_dim=None
    ):
        super().__init__()
        # Vocabulary for patch tokens
        self.vocab = vocab
        self.vocab_size = vocab.size(0)
        # Mask token (patch_dim,)
        self.mask_token = mask_token
        # Embedding and positional
        self.patch_proj = nn.Linear(patch_dim, emb_dim)
        self.pos_emb = nn.Parameter(torch.zeros(num_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        # Relative bias for attention
        self.rel_bias = nn.Parameter(torch.zeros(num_patches, num_patches))
        nn.init.trunc_normal_(self.rel_bias, std=0.02)
        # Determine latent dim if not provided
        latent_dim = latent_dim or emb_dim
        # Stacked MLA encoder layers
        self.encoder_layers = nn.ModuleList([
            MLAEncoderBlock(emb_dim, num_heads, latent_dim, ffn_dim)
            for _ in range(num_layers)
        ])
        # Final projection to vocabulary logits
        self.out_proj = nn.Linear(emb_dim, self.vocab_size)

    def forward(self, patches, mask_rate):
        # patches: [B, N, patch_dim]
        B, N, _ = patches.size()
        # 1) Random mask decision
        mask = torch.rand(B, N, device=patches.device) < mask_rate
        x = patches.clone()
        # 2) Replace masked positions with mask token
        x[mask] = self.mask_token.to(x.device)
        # 3) Project patches to embeddings + add positional
        x = self.patch_proj(x) + self.pos_emb.unsqueeze(0)
        # 4) Prepare attn bias
        attn_bias = self.rel_bias[:N, :N]
        # 5) Apply stacked MLA blocks
        for block in self.encoder_layers:
            x = block(x, attn_bias=attn_bias)
        # 6) Project to patch-token logits
        logits = self.out_proj(x)  # [B, N, vocab_size]
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
