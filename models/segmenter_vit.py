import torch
import torch.nn as nn
from typing import Tuple

class PatchEmbedding(nn.Module):
    """
    Splits the 2D spectrogram into patches and projects them into embeddings.
    """
    def __init__(self, img_size: Tuple[int, int], patch_size: int = 16, in_chans: int = 1, embed_dim: int = 768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Calculate grid size (number of patches in H and W dimensions)
        self.grid_size = (img_size[0] // patch_size, img_size[1] // patch_size)
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Use a strided convolution to extract and project patches in one step
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, C, H, W)
        returns: (B, num_patches, embed_dim), grid_size
        """
        B, C, H, W = x.shape
        
        # We compute the actual grid size for this batch dynamically
        # to support variable length audio if we choose not to pad.
        grid_size = (H // self.patch_size, W // self.patch_size)
        
        # (B, C, H, W) -> (B, embed_dim, grid_H, grid_W)
        x = self.proj(x)
        
        # Flatten the spatial dimensions: (B, embed_dim, grid_H * grid_W)
        # Then transpose: (B, num_patches, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x, grid_size

class Block(nn.Module):
    """
    A standard Transformer Block with Multi-Head Self-Attention and an MLP.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x):
        # Multi-Head Attention with residual connection
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    """
    The Vision Transformer Encoder backbone.
    """
    def __init__(self, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

class LinearDecoder(nn.Module):
    """
    A simple point-wise linear decoder that projects patch embeddings to class logits,
    then reshapes them back to a 2D spatial grid.
    """
    def __init__(self, embed_dim, num_classes):
        super().__init__()
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, grid_size):
        """
        x: (B, num_patches, embed_dim)
        grid_size: (grid_H, grid_W)
        returns: (B, num_classes, grid_H, grid_W)
        """
        # Project embeddings to classes: (B, num_patches, num_classes)
        x = self.head(x) 
        
        B, N, C = x.shape
        H, W = grid_size
        
        # Transpose and reshape: (B, num_classes, grid_H, grid_W)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x
