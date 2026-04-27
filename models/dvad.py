import torch
import torch.nn as nn
import torchvision.models.segmentation as seg_models


def create_deeplabv3(num_classes=2, pretrained_backbone=True):
    """
    Create a DeepLabV3 model with ResNet-101 backbone for binary segmentation.
    
    The DVAD paper uses DeepLabV3 as the best-performing segmentation model.
    Input: [B, 3, 512, 512] RGB spectrogram images
    Output: [B, num_classes, 512, 512] pixel-wise class logits
    
    Args:
        num_classes: Number of segmentation classes (2 for binary: noise vs clean)
        pretrained_backbone: Whether to use ImageNet-pretrained ResNet backbone
    
    Returns:
        model: DeepLabV3 model
    """
    # Use DeepLabV3 with ResNet-101 backbone
    # weights_backbone loads the ImageNet pretrained weights for the ResNet encoder
    if pretrained_backbone:
        model = seg_models.deeplabv3_resnet101(
            weights_backbone='DEFAULT',
            num_classes=num_classes
        )
    else:
        model = seg_models.deeplabv3_resnet101(
            weights_backbone=None,
            num_classes=num_classes
        )
    
    return model


class DVADSegmenter(nn.Module):
    """
    DVAD (Deep Visual Audio Denoising) Segmentation Model.
    
    Wraps DeepLabV3 to provide a clean interface for training and inference.
    The model takes RGB spectrogram images and outputs binary segmentation masks
    that identify clean audio regions vs noise regions.
    """
    def __init__(self, num_classes=2, pretrained_backbone=True):
        super().__init__()
        self.model = create_deeplabv3(
            num_classes=num_classes, 
            pretrained_backbone=pretrained_backbone
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB spectrogram images
            
        Returns:
            logits: (B, num_classes, H, W) pixel-wise class logits
        """
        output = self.model(x)
        # DeepLabV3 returns a dict with 'out' key for the main output
        return output['out']
    
    def predict_mask(self, x, threshold=0.5):
        """
        Predict binary mask for inference.
        
        Args:
            x: (B, 3, H, W) RGB spectrogram images
            threshold: Probability threshold for binary mask
            
        Returns:
            mask: (B, H, W) binary mask (1 = clean signal, 0 = noise)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            # Class 1 = clean signal
            mask = (probs[:, 1, :, :] > threshold).long()
        return mask
