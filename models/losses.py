import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation as defined in the DVAD paper (Eq. 4):
    
        Dice_loss = 1 - (2 * m ∩ m̃) / (m + m̃)
    
    where m is the ground truth mask and m̃ is the predicted mask.
    A small epsilon is added for numerical stability.
    """
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) raw logits from the segmentation model
            target: (B, H, W) ground truth binary mask (0 = noise, 1 = clean)
            
        Returns:
            dice_loss: scalar loss value
        """
        # Apply softmax to get probabilities
        pred_prob = torch.softmax(pred, dim=1)
        
        # Get the probability of the "clean" class (class index 1)
        pred_clean = pred_prob[:, 1, :, :]  # (B, H, W)
        
        # Flatten for computation
        pred_flat = pred_clean.contiguous().view(-1)
        target_flat = target.contiguous().view(-1).float()
        
        # Dice coefficient
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        
        return 1.0 - dice_coeff


class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy Loss for more stable training.
    """
    def __init__(self, dice_weight=0.5, ce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.dice_loss = DiceLoss(smooth=smooth)
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) raw logits
            target: (B, H, W) ground truth mask (long tensor)
        """
        d_loss = self.dice_loss(pred, target)
        c_loss = self.ce_loss(pred, target)
        return self.dice_weight * d_loss + self.ce_weight * c_loss
