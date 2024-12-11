#scripts/metrics
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix

def calculate_tp_fp_fn(num_classes, pred, target):
    """Calculate the true positives, false positives and false negatives for each class"""
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to('cuda')
    conf_matrix = confmat(pred, target)
    TP = conf_matrix.diag()
    FP = conf_matrix.sum(dim=0) - TP
    FN = conf_matrix.sum(dim=1) - TP

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN
    }


def plot_metric_heatmap(metric_values, metric_name, class_labels):
    """
    Generate a heatmap for per-class metric values.
    
    Args:
        metric_values (torch.Tensor): Tensor of metric values for each class.
        metric_name (str): Name of the metric (e.g., "IoU" or "Dice").
        class_labels (list of str): List of class names.
        
    Returns:
        None: Displays the heatmap.
    """
    # Convert tensor to numpy array for plotting
    metric_values = metric_values.cpu().numpy()
    
    # Create a heatmap
    sns.heatmap(metric_values[np.newaxis, :], annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_labels, yticklabels=[metric_name])
    plt.title(f"{metric_name} per Class")
    plt.xlabel("Classes")

def calculate_iou_per_class(results):
    """
    Compute the IoU for each class.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        iou_per_class (torch.Tensor): IoU values for each class.
    """
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    
    # Compute IoU for each class
    iou_per_class = TP / (TP + FP + FN + 1e-7)  # Add epsilon to avoid division by zero
    return iou_per_class

# mean iou
def calculate_mean_iou(results):
    """
    Compute the mean Intersection over Union (IoU) across all classes.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        mean_iou (torch.Tensor): Mean IoU across all classes.
    """
    iou_per_class = calculate_iou_per_class(results)
    mean_iou = iou_per_class.mean()
    return mean_iou



# dice Similarity Coefficient
def calculate_dice_per_class(results):
    """
    Compute the Dice Similarity Coefficient for each class.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        dice_per_class (torch.Tensor): Dice values for each class.
    """
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    
    # Compute Dice for each class
    dice_per_class = 2 * TP / (2 * TP + FP + FN + 1e-7)  # Add epsilon to avoid division by zero
    return dice_per_class

# mean dice 
def calculate_mean_dice(results):
    """
    Compute the mean Dice Similarity Coefficient across all classes.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        mean_dice (torch.Tensor): Mean Dice across all classes.
    """
    dice_per_class = calculate_dice_per_class(results)
    mean_dice = dice_per_class.mean()
    return mean_dice


# mean pixel accuracy
def calculate_mean_pixel_accuracy(results):
    """
    Compute the mean pixel accuracy across all classes.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        mean_pixel_accuracy (torch.Tensor): Mean pixel accuracy across all classes.
    """
    TP = results["TP"]
    FP = results["FP"]
    
    # Compute mean pixel accuracy
    mean_pixel_accuracy = TP.sum() / (TP.sum() + FP.sum() + 1e-7)  # Add epsilon to avoid division by zero
    return mean_pixel_accuracy


## Confusion Matrix (per class) for visualization

## 	Class-wise IoU or Dice Heatmap

