import torch
import torch.nn as nn
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F
import numpy as np


class WeightedSegmentationLoss(nn.Module):
    def __init__(self, num_classes=6, sigma=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma
        
    def create_background_weights(self, mask):
        mask_np = mask.cpu().numpy()
        batch_weights = []

        for batch_idx in range(mask_np.shape[0]):
            # Create binary mask of non-background pixels (everything that's not class 5)
            non_background = mask_np[batch_idx] != 5
            
            # Calculate distance transform
            distances = distance_transform_edt(~non_background)
            weights = np.exp(-distances**2 / (2 * self.sigma**2))
            
            batch_weights.append(weights)
        
        # Stack weights and convert back to tensor
        batch_weights = np.stack(batch_weights)
        return torch.tensor(batch_weights, dtype=torch.float32).to(mask.device)
    
    def focal_loss(self, pred, target, gamma=2.0, alpha=0.25):
        """
        Compute Focal Loss to handle class imbalance.
        Args:
            pred: Predicted probabilities for each class (shape: [batch_size, num_classes, height, width])
            target: True class indices (shape: [batch_size, height, width])
            gamma: Focusing parameter to adjust the rate at which easy examples are down-weighted
            alpha: Weight factor to balance class distribution
        """
        target_onehot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Predicted probabilities for each class
        pred_softmax = F.softmax(pred, dim=1)
        cross_entropy = -target_onehot * torch.log(pred_softmax + 1e-6)

        # Compute modulating factor (1 - p_t)^gamma
        modulating_factor = (1 - pred_softmax) ** gamma

        # Apply focal loss modulation
        focal_loss = alpha * modulating_factor * cross_entropy

        # Return the mean loss for each pixel
        return focal_loss.sum(dim=(2, 3)).mean()

    def dice_loss(self, pred, target, smooth=1e-6):
        """
        Compute Dice loss between prediction and target tensors.
        
        Args:
            pred: Tensor of shape [batch_size, num_classes, height, width] - Predicted probabilities
            target: Tensor of shape [batch_size, height, width] - True class indices
            smooth: Small constant to avoid division by zero
        
        Returns:
            dice_loss: The calculated Dice loss
        """
        # One-hot encode the target tensor
        target_onehot = torch.nn.functional.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Predicted probabilities for each class (already softmaxed in the output)
        pred_softmax = F.softmax(pred, dim=1)

        intersection = torch.sum(pred_softmax * target_onehot, dim=(2, 3))
        union = torch.sum(pred_softmax + target_onehot, dim=(2, 3))
        
        # Compute Dice coefficient
        dice = (2 * intersection + smooth) / (union + smooth)
        
        # Return Dice loss
        return 1 - dice.mean()  # We want to minimize Dice loss, so 1 - Dice coefficient
    
    def forward(self, pred, target):
        """
        Args:
            pred: Tensor of shape [batch_size, num_classes, height, width]
            target: Tensor of shape [batch_size, height, width] containing class indices
        Returns:
            Total loss combining cross-entropy (with focal loss) and Dice loss
        """
        # Get distance-based weights that emphasize background pixels near class boundaries
        background_weights = self.create_background_weights(target)  # [batch_size, height, width]
        
        # Calculate standard pixel loss (Focal or Cross-Entropy loss)
        pixel_losses = self.focal_loss(pred, target)  # Use focal loss instead of regular cross-entropy

        # Calculate Dice loss
        dice_loss = self.dice_loss(pred, target)

        # Combine the weighted pixel losses and Dice loss (adjust lambda for balance)
        total_loss = pixel_losses + dice_loss  # Adjust lambda if needed

        return total_loss