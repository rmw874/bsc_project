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
        epsilon = 1e-5  # Small constant to ensure non-zero weights

        for batch_idx in range(mask_np.shape[0]):
            # Create binary mask of non-background pixels (everything that's not class 5)
            non_background = mask_np[batch_idx] != 5
            # Calculate distance transform
            distances = distance_transform_edt(~non_background)
            # Calculate weights and ensure non-zero with epsilon
            weights = np.exp(-distances**2 / (2 * self.sigma**2)) + 1 # changed from epilson to 1
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
        pred_softmax = torch.clamp(pred_softmax, min=1e-6, max=1.0)
        cross_entropy = -target_onehot * torch.log(pred_softmax + 1e-6)

        # Compute modulating factor (1 - p_t)^gamma
        modulating_factor = (1 - pred_softmax) ** gamma

        # Apply focal loss modulation
        focal_loss = alpha * modulating_factor * cross_entropy

        # Return the mean loss for each pixel
        return focal_loss.sum(dim=(2, 3)).mean(dim=0)
    
    def focal_loss_with_weights(self, pred, target, gamma=2.0, alpha=0.25):
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
        pred_softmax = torch.clamp(pred_softmax, min=1e-6, max=1.0)
        cross_entropy = -target_onehot * torch.log(pred_softmax + 1e-6)
        cross_entropy_weighted = cross_entropy * self.create_background_weights(target).unsqueeze(1)

        # Compute modulating factor (1 - p_t)^gamma
        modulating_factor = (1 - pred_softmax) ** gamma

        # Apply focal loss modulation
        focal_loss = alpha * modulating_factor * cross_entropy_weighted

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
    
    def forward(self, pred, target, lambda_focal=1.0, lambda_dice=1.0):
        """
        Args:
            pred: Tensor of shape [batch_size, num_classes, height, width]
            target: Tensor of shape [batch_size, height, width] containing class indices
            lambda_focal: Weight for the focal loss term
            lambda_dice: Weight for the dice loss term
        Returns:
            Total loss combining weighted Focal and Dice loss
        """
        # Calculate Focal Loss with background weights
        pixel_loss = self.focal_loss(pred, target)
        
        #pixel_loss_weighted = self.focal_loss_with_weights(pred, target)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Combine losses with dynamic weights
        total_loss = lambda_focal * pixel_loss + lambda_dice * dice_loss

        return total_loss


def loss_weight_scheduler(epoch, total_epochs, switch_epoch=15):
    """
    Schedule weights for Focal Loss and Dice Loss dynamically.
    
    Args:
        epoch: Current epoch number
        total_epochs: Total number of epochs for training
        switch_epoch: Epoch after which Dice Loss importance increases
    
    Returns:
        lambda_focal: Weight for Focal Loss
        lambda_dice: Weight for Dice Loss
    """
    if epoch < switch_epoch:
        # Focus on Focal Loss in the early phase
        lambda_focal = 0.5
        lambda_dice = 0.5
    else:
        # Gradually shift focus to Dice Loss after switch_epoch
        progress = (epoch - switch_epoch) / (total_epochs - switch_epoch)
        lambda_focal = 0.5 - 0.4 * progress  # Decrease Focal Loss weight
        lambda_dice = 0.5 + 0.4 * progress         # Increase Dice Loss weight
    
    return lambda_focal, lambda_dice


### CONSIDER USING TVERSKEY FOCAL LOSS INSTEAD OF DICE LOSS COMBINED WITH FOCAL LOSS
### CONSIDER MAKING IMAGES BE WHITE WHERE THERE IS NOTHING OF INTEREST
### CONSIDER MAKING WEIGHT DECAY
### CONSIDER SMALL UNET MODEL
### CONSIDER USING ALBUMENTATIONS FOR AUGMENTATION
### CONSIDER USING REGULARRIZATION