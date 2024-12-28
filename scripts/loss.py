import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch.losses import TverskyLoss
import numpy as np
from scipy.ndimage import distance_transform_edt
from kornia.filters import sobel

class WeightedSegmentationLoss(nn.Module):
    def __init__(self, num_classes=6, sigma=5.0, use_tversky=False, tversky_alpha=0.5, tversky_beta=0.5, from_logits=True):
        super().__init__()
        self.num_classes = num_classes
        self.sigma = sigma

        # Initialize TverskyLoss if desired
        self.use_tversky = use_tversky
        if self.use_tversky:
            # mode='multiclass' handles multiple classes
            self.tversky_loss_fn = TverskyLoss(
                mode='multiclass', 
                from_logits=from_logits, 
                alpha=tversky_alpha, 
                beta=tversky_beta
            )
    
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
        # pred_softmax = torch.clamp(pred_softmax, min=1e-6, max=1.0)
        cross_entropy = -target_onehot * torch.log(pred_softmax + 1e-6)

        # Compute modulating factor (1 - p_t)^gamma
        modulating_factor = (1 - pred_softmax) ** gamma

        # Apply focal loss modulation
        focal_loss = alpha * modulating_factor * cross_entropy

        # Return the mean loss for each pixel
        return focal_loss.sum(dim=(2, 3)).mean()
    
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

        # If TverskyLoss is enabled, just return that directly to test how it works
        if self.use_tversky:
            # Use TverskyLoss directly:
            # pred: [B, C, H, W] raw logits
            # target: [B, H, W] class indices
            return self.tversky_loss_fn(pred, target)
        

        # Calculate Focal Loss with background weights
        pixel_loss = self.focal_loss(pred, target)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(pred, target)

        # Combine losses with dynamic weights
        total_loss = lambda_focal * pixel_loss + lambda_dice * dice_loss

        return total_loss

# class BoundaryAwareTverskyLoss(nn.Module):
#     def __init__(self, num_classes=6, alpha=0.5, beta=0.5, smooth=1e-6,
#                  boundary_weight=1.0, from_logits=True, class_weights=None, sigma=10.0):
#         super().__init__()
#         self.num_classes = num_classes
#         self.alpha = alpha
#         self.beta = beta
#         self.smooth = smooth
#         self.from_logits = from_logits
#         self.sigma = sigma
#         self.boundary_weight = boundary_weight
#         self.background_class = 5
        
#         if class_weights is None:
#             self.class_weights = torch.ones(num_classes)
#         else:
#             self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

#     def compute_boundary_maps(self, target):
#         """Compute boundary maps for each class"""
#         target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
#         boundaries = torch.zeros_like(target_onehot)
        
#         for c in range(self.num_classes):
#             # Get binary mask for current class
#             class_mask = target_onehot[:, c:c+1]
            
#             # Compute gradients to detect edges
#             edges = torch.abs(sobel(class_mask))
#             boundaries[:, c:c+1] = (edges > 0.5).float()
        
#         return boundaries

#     def create_background_weights(self, mask):
#         """Create distance-based weights giving more importance to pixels near objects"""
#         mask_np = mask.cpu().numpy()
#         batch_weights = []
        
#         for batch_idx in range(mask_np.shape[0]):
#             non_background = mask_np[batch_idx] != self.background_class
#             distances = distance_transform_edt(~non_background)
#             weights = np.exp(-distances**2 / (2 * self.sigma**2)) + 1
#             batch_weights.append(weights)

#         batch_weights = np.stack(batch_weights)
#         return torch.tensor(batch_weights, dtype=torch.float32).to(mask.device)

#     def forward(self, pred, target):
#         device = pred.device
#         self.class_weights = self.class_weights.to(device)

#         if self.from_logits:
#             pred = F.softmax(pred, dim=1)
        
#         # Get background weights and boundary maps
#         background_weights = self.create_background_weights(target)
#         boundary_maps = self.compute_boundary_maps(target)
        
#         # Normalize the combined weights to prevent inflation
#         combined_weights = background_weights.unsqueeze(1) * (1 + self.boundary_weight * boundary_maps)
#         combined_weights = combined_weights / combined_weights.mean()
        
#         # One-hot encode target
#         target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

#         # Compute intersection and union with weights
#         intersection = torch.sum(pred * target_onehot * combined_weights, dim=(0,2,3))
#         fps = torch.sum(pred * (1 - target_onehot) * combined_weights, dim=(0,2,3))
#         fns = torch.sum((1 - pred) * target_onehot * combined_weights, dim=(0,2,3))

#         # Compute Tversky index
#         numerator = intersection + self.smooth
#         denominator = intersection + self.alpha * fns + self.beta * fps + self.smooth
#         tversky_per_class = numerator / denominator

#         # Ensure loss is positive by using log formulation
#         loss = -torch.log(torch.mean(self.class_weights * tversky_per_class) + self.smooth)
        
#         return loss

class BoundaryAwareTverskyLoss(nn.Module):
    def __init__(self, num_classes=6, alpha=0.4, beta=0.6, smooth=1e-6,
                 boundary_weight=2.5, vertical_consistency_weight=1.0,
                 from_logits=True, class_weights=None, sigma=5.0):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        self.sigma = sigma
        self.boundary_weight = boundary_weight
        self.vertical_consistency_weight = vertical_consistency_weight
        self.background_class = 5
        
        if class_weights is None:
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_boundary_maps(self, target):
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        boundaries = torch.zeros_like(target_onehot)
        
        for c in range(self.num_classes):
            class_mask = target_onehot[:, c:c+1]
            edges = torch.abs(sobel(class_mask))
            threshold = edges.mean() + edges.std()
            boundaries[:, c:c+1] = (edges > threshold).float()
        
        return boundaries

    def compute_vertical_consistency_loss(self, pred):
        vertical_diff = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        horizontal_diff = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        return horizontal_diff.mean() * 2.0 - vertical_diff.mean()

    def compute_horizontal_consistency_loss(self, pred):
        horizontal_diff = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        return horizontal_diff.mean()

    def forward(self, pred, target):
        device = pred.device
        self.class_weights = self.class_weights.to(device)

        if self.from_logits:
            pred_softmax = F.softmax(pred, dim=1)
        else:
            pred_softmax = pred

        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Compute boundary weights
        edges = torch.zeros_like(pred_softmax)
        for c in range(self.num_classes):
            class_mask = target_onehot[:, c:c+1]
            edges[:, c:c+1] = torch.abs(sobel(class_mask))
        
        boundary_weights = (edges > edges.mean()).float() * self.boundary_weight * 2.0 + 1.0
        
        # Compute weighted metrics
        intersection = torch.sum(pred_softmax * target_onehot * boundary_weights, dim=(0,2,3))
        fps = torch.sum(pred_softmax * (1 - target_onehot) * boundary_weights, dim=(0,2,3))
        fns = torch.sum((1 - pred_softmax) * target_onehot * boundary_weights, dim=(0,2,3))

        tversky_per_class = (intersection + self.smooth) / (intersection + self.alpha * fns + self.beta * fps + self.smooth)
        
        # Add horizontal consistency
        horizontal_loss = self.compute_horizontal_consistency_loss(pred_softmax)

        vertical_loss = self.compute_vertical_consistency_loss(pred_softmax)
        tversky_loss = -torch.log(torch.mean(self.class_weights * tversky_per_class) + self.smooth)
        
        return tversky_loss + self.vertical_consistency_weight * vertical_loss + horizontal_loss



def loss_weight_scheduler(epoch, total_epochs, switch_epoch=100):
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
