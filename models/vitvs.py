import torch
import torch.nn as nn
import torch.nn.functional as F

from .segmenter_vit import PatchEmbedding, TransformerEncoder, LinearDecoder

class ViTVS_Segmenter(nn.Module):
    """
    Vision Transformer for Vocal Separation (ViTVS) - Image Segmentation Backbone.
    Based on the paper "Vision Transformer Segmentation for Visual Bird Sound Denoising".
    
    This model takes RGB audio images (spectrograms) as input and outputs a binary
    segmentation mask (clean audio vs noise).
    """
    def __init__(self, 
                 img_size=(256, 256), 
                 patch_size=16, 
                 in_chans=3, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 num_classes=2):
        super().__init__()
        
        # 1. Image to Patches & Embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=in_chans, 
            embed_dim=embed_dim
        )
        
        # Absolute positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        # 2. Vision Transformer Encoder (12 blocks)
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads
        )
        
        # 3. Linear Decoder
        self.decoder = LinearDecoder(
            embed_dim=embed_dim, 
            num_classes=num_classes
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 256, 256) RGB audio images
            
        Returns:
            logits: (B, 2, 256, 256) segmentation logits
        """
        # Extract Patches
        x_embed, grid_size = self.patch_embed(x)
        
        # Add positional embedding
        if x_embed.shape[1] != self.pos_embed.shape[1]:
            orig_grid = self.patch_embed.grid_size
            pe = self.pos_embed.transpose(1, 2).view(1, -1, orig_grid[0], orig_grid[1])
            pe = F.interpolate(pe, size=grid_size, mode='bilinear', align_corners=False)
            pe = pe.flatten(2).transpose(1, 2)
            x_embed = x_embed + pe
        else:
            x_embed = x_embed + self.pos_embed
            
        # Encoder
        encoded = self.encoder(x_embed)
        
        # Decoder
        logits = self.decoder(encoded, grid_size)
        
        # Upsample back to original image size (256x256)
        # because the decoder outputs at grid resolution (e.g. 16x16)
        logits_upsampled = F.interpolate(
            logits, 
            size=x.shape[2:], 
            mode='bilinear', 
            align_corners=False
        )
        
        return logits_upsampled
