import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_processing import wav_to_spectrogram, spectrogram_to_wav
from .segmenter_vit import PatchEmbedding, TransformerEncoder, LinearDecoder

class ViTVS(nn.Module):
    """
    Vision Transformer for Vocal Separation (ViTVS).
    Combines STFT, Segmenter ViT backbone, and ISTFT for audio denoising.
    """
    def __init__(self, 
                 img_size=(128, 128), # Default expected spectrogram size
                 patch_size=16, 
                 embed_dim=768, 
                 depth=12, 
                 num_heads=12, 
                 num_classes=2,
                 n_fft=1024, 
                 hop_length=256, 
                 win_length=1024):
        super().__init__()
        
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.patch_size = patch_size
        
        # 1. Segmenter Components
        self.patch_embed = PatchEmbedding(
            img_size=img_size, 
            patch_size=patch_size, 
            in_chans=1, 
            embed_dim=embed_dim
        )
        
        # Absolute positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        
        self.encoder = TransformerEncoder(
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads
        )
        
        self.decoder = LinearDecoder(
            embed_dim=embed_dim, 
            num_classes=num_classes
        )

    def forward(self, wav):
        """
        Args:
            wav: (B, T) audio waveform
            
        Returns:
            denoised_wav: (B, T) reconstructed waveform
            logits_upsampled: (B, num_classes, F, T_frames) for calculating NLL loss
        """
        # 1. Audio to Spectrogram
        magnitude, phase = wav_to_spectrogram(
            wav, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length
        )
        
        # magnitude is (B, 1, F, T_frames)
        B, C, F, T_frames = magnitude.shape
        
        # Pad dimensions to be divisible by patch_size
        pad_F = (self.patch_size - F % self.patch_size) % self.patch_size
        pad_T = (self.patch_size - T_frames % self.patch_size) % self.patch_size
        
        if pad_F > 0 or pad_T > 0:
            magnitude_padded = F.pad(magnitude, (0, pad_T, 0, pad_F), mode='constant', value=0)
        else:
            magnitude_padded = magnitude
            
        # 2. Extract Patches
        x, grid_size = self.patch_embed(magnitude_padded)
        
        # Add positional embedding. 
        # Interpolate pos_embed if the input spectrogram size differs from the default img_size
        if x.shape[1] != self.pos_embed.shape[1]:
            orig_grid = self.patch_embed.grid_size
            pe = self.pos_embed.transpose(1, 2).view(1, -1, orig_grid[0], orig_grid[1])
            pe = F.interpolate(pe, size=grid_size, mode='bilinear', align_corners=False)
            pe = pe.flatten(2).transpose(1, 2)
            x = x + pe
        else:
            x = x + self.pos_embed
            
        # 3. Vision Transformer Encoder
        x = self.encoder(x)
        
        # 4. Decoder
        logits = self.decoder(x, grid_size)
        
        # The decoder output is at the patch grid resolution (grid_H, grid_W).
        # We upsample it to match the padded magnitude spectrogram size (F + pad_F, T_frames + pad_T).
        logits_upsampled = F.interpolate(
            logits, 
            size=(F + pad_F, T_frames + pad_T), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Crop back to original spectrogram size if it was padded
        if pad_F > 0 or pad_T > 0:
            logits_upsampled = logits_upsampled[:, :, :F, :T_frames]
            
        # Apply Softmax to get probabilities
        probs = F.softmax(logits_upsampled, dim=1)
        
        # Class 1 is assumed to be the "Bird" / "Clean Audio" class.
        mask = probs[:, 1:2, :, :] # (B, 1, F, T_frames)
        
        # Apply the predicted mask to the magnitude spectrogram
        masked_magnitude = magnitude * mask
        
        # 5. Spectrogram to Audio (ISTFT)
        denoised_wav = spectrogram_to_wav(
            masked_magnitude, 
            phase, 
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length
        )
        
        # Ensure denoised wav length matches input wav length exactly
        if denoised_wav.shape[-1] > wav.shape[-1]:
            denoised_wav = denoised_wav[..., :wav.shape[-1]]
        elif denoised_wav.shape[-1] < wav.shape[-1]:
            pad_val = wav.shape[-1] - denoised_wav.shape[-1]
            denoised_wav = F.pad(denoised_wav, (0, pad_val))
            
        return denoised_wav, logits_upsampled
