import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

class HausdorffDistanceLoss(nn.Module):
    def __init__(self, alpha=1.0):
        """
        Hausdorff Distance Loss for segmentation.

        Args:
            alpha (float): Weighting factor for the distance.
        """
        super().__init__()
        self.alpha = alpha

    @staticmethod
    def compute_distance_map(binary_mask):
        """
        Compute the distance map for a binary mask.
        
        Args:
            binary_mask: Binary mask of shape [B, H, W].
        
        Returns:
            Distance map of shape [B, H, W].
        """
        batch_size = binary_mask.shape[0]
        distance_maps = []
        for i in range(batch_size):
            distance_map = distance_transform_edt(1 - binary_mask[i].cpu().numpy())
            distance_maps.append(distance_map)
        return torch.tensor(distance_maps, dtype=torch.float32).to(binary_mask.device)

    def forward(self, pred, target):
        """
        Compute Hausdorff Distance Loss.

        Args:
            pred: Predicted probabilities or logits, shape [B, C, H, W].
            target: Ground truth segmentation, shape [B, H, W].

        Returns:
            Hausdorff Distance Loss (scalar).
        """
        # Convert predictions to binary masks using argmax
        pred = torch.argmax(pred, dim=1)  # Shape: [B, H, W]

        # Compute distance maps
        pred_distance_map = self.compute_distance_map(pred)
        target_distance_map = self.compute_distance_map(target)

        # Compute directed Hausdorff distances
        hd_pred_to_target = torch.mean(pred * target_distance_map)
        hd_target_to_pred = torch.mean(target * pred_distance_map)

        # Combine distances to get symmetric Hausdorff Distance
        hausdorff_distance = (hd_pred_to_target + hd_target_to_pred) / 2.0

        return self.alpha * hausdorff_distance
