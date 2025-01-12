import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import sobel

class TverskyLoss(nn.Module):
    def __init__(self, num_classes=6, alpha=0.5, beta=0.5, smooth=1e-6, from_logits=True):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = F.softmax(pred, dim=1)
        
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred * target_onehot, dim=(0,2,3))
        fps = torch.sum(pred * (1 - target_onehot), dim=(0,2,3))
        fns = torch.sum((1 - pred) * target_onehot, dim=(0,2,3))
        
        numerator = intersection + self.smooth
        denominator = intersection + self.alpha * fns + self.beta * fps + self.smooth
        tversky_per_class = numerator / denominator
        loss = -torch.log(torch.mean(tversky_per_class) + self.smooth)
        
        return loss

class DiceLoss(nn.Module):
    def __init__(self, num_classes=6, smooth=1e-6, from_logits=True):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.from_logits = from_logits
    
    def forward(self, pred, target):
        if self.from_logits:
            pred = F.softmax(pred, dim=1)
        
        target_one_hot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = torch.sum(pred * target_one_hot, dim=(2, 3))
        cardinality = torch.sum(pred + target_one_hot, dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1 - dice.mean()
        
        return dice_loss

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
        
        # compute boundary weights
        edges = torch.zeros_like(pred_softmax)
        for c in range(self.num_classes):
            class_mask = target_onehot[:, c:c+1]
            edges[:, c:c+1] = torch.abs(sobel(class_mask))
        
        boundary_weights = (edges > edges.mean()).float() * self.boundary_weight * 2.0 + 1.0
        
        # compute weighted metrics
        intersection = torch.sum(pred_softmax * target_onehot * boundary_weights, dim=(0,2,3))
        fps = torch.sum(pred_softmax * (1 - target_onehot) * boundary_weights, dim=(0,2,3))
        fns = torch.sum((1 - pred_softmax) * target_onehot * boundary_weights, dim=(0,2,3))

        tversky_per_class = (intersection + self.smooth) / (intersection + self.alpha * fns + self.beta * fps + self.smooth)
        
        # add horizontal consistency
        horizontal_loss = self.compute_horizontal_consistency_loss(pred_softmax)

        vertical_loss = self.compute_vertical_consistency_loss(pred_softmax)
        tversky_loss = -torch.log(torch.mean(self.class_weights * tversky_per_class) + self.smooth)
        
        return tversky_loss + self.vertical_consistency_weight * vertical_loss + horizontal_loss

class FocalLoss(nn.Module):
    def __init__(self, num_classes=6, gamma=2.0, alpha=0.25, smooth=1e-6, from_logits=True):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = smooth
        self.from_logits = from_logits

    def forward(self, pred, target):
        if self.from_logits:
            pred = F.softmax(pred, dim=1)
        
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        pred = torch.clamp(pred, min=self.smooth, max=1.0 - self.smooth)
        
        focal_weight = (1 - pred) ** self.gamma
        ce_loss = -target_onehot * torch.log(pred)
        focal_loss = self.alpha * focal_weight * ce_loss
        
        return focal_loss.sum(dim=(2, 3)).mean()

class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=6, smooth=1e-6, from_logits=True, class_weights=None):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        self.from_logits = from_logits
        self.class_weights = class_weights
        
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def forward(self, pred, target):
        if self.from_logits:
            pred = F.softmax(pred, dim=1)
        
        target_onehot = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        pred = torch.clamp(pred, min=self.smooth, max=1.0 - self.smooth)
        
        ce_loss = -target_onehot * torch.log(pred)
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(pred.device)
            ce_loss = ce_loss * self.class_weights.view(1, -1, 1, 1)
        
        return ce_loss.sum(dim=(2, 3)).mean()

class WeightedCompositeLoss(nn.Module):
    def __init__(self, num_classes=6, focal_gamma=2.0, focal_alpha=0.25, 
                 dice_smooth=1e-6, from_logits=True, class_weights=None):
        super().__init__()
        self.focal_loss = FocalLoss(
            num_classes=num_classes,
            gamma=focal_gamma,
            alpha=focal_alpha,
            from_logits=from_logits
        )
        self.dice_loss = DiceLoss(
            num_classes=num_classes,
            smooth=dice_smooth,
            from_logits=from_logits
        )
        self.class_weights = class_weights

    def forward(self, pred, target, lambda_focal=0.5, lambda_dice=0.5):
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(pred.device)
            focal_loss = focal_loss * self.class_weights.mean()
            dice_loss = dice_loss * self.class_weights.mean()
        
        return lambda_focal * focal_loss + lambda_dice * dice_loss

    def update_lambdas(self, epoch, total_epochs, switch_epoch=100):
        """
        Update lambda weights based on training progress.
        """
        if epoch < switch_epoch:
            lambda_focal = 0.5
            lambda_dice = 0.5
        else:
            progress = (epoch - switch_epoch) / (total_epochs - switch_epoch)
            lambda_focal = 0.5 - 0.4 * progress
            lambda_dice = 0.5 + 0.4 * progress
        
        return lambda_focal, lambda_dice