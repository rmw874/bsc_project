import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedTverskyLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.5, beta=0.5, smooth=1e-6, from_logits=True, class_weights=None):
        """
        Weighted Tversky Loss
        
        Args:
            num_classes (int): Number of segmentation classes.
            alpha (float): Controls penalty for FN.
            beta (float): Controls penalty for FP.
            smooth (float): Smooth factor to avoid division by zero.
            from_logits (bool): If True, 'pred' are raw logits and will be passed through softmax.
            class_weights (list or torch.Tensor): Weights for each class. 
                If None, all classes are equally weighted.
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits
        
        if class_weights is None:
            # If no weights provided, default to uniform weights
            self.class_weights = torch.ones(num_classes)
        else:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, pred, target):
        """
        Args:
            pred: [B, C, H, W] raw logits or probabilities
            target: [B, H, W] class indices
        Returns:
            Weighted Tversky Loss (scalar)
        """
        device = pred.device
        self.class_weights = self.class_weights.to(device)

        if self.from_logits:
            # Convert logits to probabilities
            pred = F.softmax(pred, dim=1)
        
        # One-hot encode the target
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()

        # Compute per-class Tversky
        # TP: sum of pred * target for each class
        TP = torch.sum(pred * target_onehot, dim=(0,2,3))
        FP = torch.sum(pred * (1 - target_onehot), dim=(0,2,3))
        FN = torch.sum((1 - pred) * target_onehot, dim=(0,2,3))

        tversky_per_class = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)

        # Apply class weights. We want a loss, so we use (1 - Tversky).
        # Weighted mean across classes
        weighted_tversky = torch.sum(self.class_weights * (1 - tversky_per_class)) / torch.sum(self.class_weights)

        return weighted_tversky