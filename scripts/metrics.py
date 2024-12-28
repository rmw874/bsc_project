#scripts/metrics
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from torchmetrics.classification import MulticlassConfusionMatrix
import torch.nn.functional as F
from skimage.measure import label
from kornia.filters import sobel

epsilon = 1e-7

def calculate_confusion_metrics(num_classes, pred, target):
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to('cuda')
    conf_matrix = confmat(pred, target)
    TP = conf_matrix.diag()
    FP = conf_matrix.sum(dim=0) - TP
    FN = conf_matrix.sum(dim=1) - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "TN": TN,
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
    metric_values = metric_values.cpu().numpy()
    
    # Create a heatmap
    sns.heatmap(metric_values[np.newaxis, :], annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_labels, yticklabels=[metric_name])
    # plt.title(f"{metric_name} per Class")
    plt.title("")
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
    
    iou_per_class = TP / (TP + FP + FN + epsilon)
    return iou_per_class

def calculate_mean_iou(results):
    """
    Compute the mean Intersection over Union (IoU) across all classes, 
    excluding classes absent in the ground truth.
    
    Args:
        results (dict): Dictionary containing "TP", "FP", and "FN" as tensors of shape [num_classes].
    
    Returns:
        mean_iou (torch.Tensor): Mean IoU across all present classes.
    """
    iou_per_class = calculate_iou_per_class(results)
    
    # Identify classes with ground truth (TP + FN > 0)
    valid_classes = (results["TP"] + results["FN"]) > 0
    
    # Calculate the mean IoU for valid classes only
    mean_iou = iou_per_class[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
    return mean_iou

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
    
    dice_per_class = 2 * TP / (2 * TP + FP + FN + epsilon)
    return dice_per_class

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
    
    mean_pixel_accuracy = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    return mean_pixel_accuracy

def calculate_accuracy(results):
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    TN = results["TN"]
    
    # Compute accuracy as sum of TP + TN over total elements (TP + FP + FN + TN for all classes)
    total_correct = (TP + TN).sum()  # Sum of TP + TN across all classes
    total = (TP + FP + FN + TN).sum()  # Total elements (for all classes)
    
    accuracy = total_correct / total  # Accuracy formula
    return accuracy

def calculate_precision_per_class(results):
    TP = results["TP"]
    FP = results["FP"]

    return TP / (TP + FP + 1e-7)

def calculate_mean_precision(results):
    precision_per_class = calculate_precision_per_class(results)
    return precision_per_class.mean()

def calculate_recall_per_class(results):
    TP = results["TP"]
    FN = results["FN"]
    
    return TP / (TP + FN + 1e-7)

def calculate_mean_recall(results):
    recall_per_class = calculate_recall_per_class(results)
    return recall_per_class.mean()

def calculate_boundary_metrics(pred, target, num_classes):
    """Calculate boundary-specific metrics with improved boundary detection"""
    pred_softmax = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_softmax, dim=1)
    
    # Convert to one-hot
    pred_onehot = F.one_hot(pred_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    # Compute boundaries with multi-scale approach
    pred_boundaries = torch.zeros_like(pred_onehot)
    target_boundaries = torch.zeros_like(target_onehot)
    
    for c in range(num_classes):
        # Compute edges with Sobel
        pred_edges = torch.abs(sobel(pred_onehot[:, c:c+1]))
        target_edges = torch.abs(sobel(target_onehot[:, c:c+1]))
        
        # Use adaptive thresholding for boundary detection
        pred_threshold = pred_edges.mean() + pred_edges.std()
        target_threshold = target_edges.mean() + target_edges.std()
        
        pred_boundaries[:, c:c+1] = (pred_edges > pred_threshold).float()
        target_boundaries[:, c:c+1] = (target_edges > target_threshold).float()
    
    # Calculate metrics only for classes present in target
    boundary_metrics = {'boundary_precision': 0.0, 'boundary_recall': 0.0, 'boundary_f1': 0.0}
    valid_classes = 0
    
    for c in range(num_classes):
        if target_boundaries[:, c].sum() > 0:  # Only consider classes with boundaries in target
            tp = torch.sum(pred_boundaries[:, c] * target_boundaries[:, c])
            fp = torch.sum(pred_boundaries[:, c] * (1 - target_boundaries[:, c]))
            fn = torch.sum((1 - pred_boundaries[:, c]) * target_boundaries[:, c])
            
            precision = tp / (tp + fp + epsilon)
            recall = tp / (tp + fn + epsilon)
            f1 = 2 * precision * recall / (precision + recall + epsilon)
            
            boundary_metrics['boundary_precision'] += precision
            boundary_metrics['boundary_recall'] += recall
            boundary_metrics['boundary_f1'] += f1
            valid_classes += 1
    
    if valid_classes > 0:
        boundary_metrics = {k: v/valid_classes for k, v in boundary_metrics.items()}
    
    return boundary_metrics

def calculate_instance_separation(pred, target, num_classes):
    """Calculate metrics for instance separation"""
    pred_softmax = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_softmax, dim=1)
    separation_scores = []
    
    for b in range(pred_labels.shape[0]):
        batch_scores = []
        for c in range(num_classes):
            # Get binary masks for current class
            pred_mask = (pred_labels[b] == c).cpu().numpy()
            target_mask = (target[b] == c).cpu().numpy()
            
            # Label connected components
            pred_instances = label(pred_mask)
            target_instances = label(target_mask)
            
            # Only calculate score if the class exists in the target
            if target_instances.max() > 0:
                # Calculate the number of predicted and target instances
                pred_instance_count = len(np.unique(pred_instances)) - 1  # subtract 1 for background
                target_instance_count = len(np.unique(target_instances)) - 1
                
                # Calculate separation score (bounded between 0 and 1)
                if target_instance_count == 0:
                    score = 1.0 if pred_instance_count == 0 else 0.0
                else:
                    score = max(0.0, min(1.0, 1.0 - abs(pred_instance_count - target_instance_count) / target_instance_count))
                batch_scores.append(score)
        
        if batch_scores:
            separation_scores.append(np.mean(batch_scores))
    
    return torch.tensor(np.mean(separation_scores) if separation_scores else 0.0).to(pred.device)