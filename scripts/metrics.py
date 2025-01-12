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
    device = pred.device
    confmat = MulticlassConfusionMatrix(num_classes=num_classes).to(device)
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
    metric_values = metric_values.cpu().numpy()
    
    sns.heatmap(metric_values[np.newaxis, :], annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=class_labels, yticklabels=[metric_name])
    plt.title(" ")
    plt.xlabel("Classes")

def calculate_iou_per_class(results):
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    
    iou_per_class = TP / (TP + FP + FN + epsilon)
    return iou_per_class

def calculate_mean_iou(results):
    iou_per_class = calculate_iou_per_class(results)
    
    # Identify classes with ground truth (TP + FN > 0)
    valid_classes = (results["TP"] + results["FN"]) > 0
    
    # Calculate the mean IoU for valid classes only
    mean_iou = iou_per_class[valid_classes].mean() if valid_classes.any() else torch.tensor(0.0)
    return mean_iou

def calculate_dice_per_class(results):
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    
    dice_per_class = 2 * TP / (2 * TP + FP + FN + epsilon)
    return dice_per_class

def calculate_mean_dice(results):
    dice_per_class = calculate_dice_per_class(results)
    mean_dice = dice_per_class.mean()
    return mean_dice

def calculate_mean_pixel_accuracy(results):
    TP = results["TP"]
    FP = results["FP"]
    
    mean_pixel_accuracy = TP.sum() / (TP.sum() + FP.sum() + epsilon)
    return mean_pixel_accuracy

def calculate_accuracy(results):
    TP = results["TP"]
    FP = results["FP"]
    FN = results["FN"]
    TN = results["TN"]
    
    total_correct = (TP + TN).sum()  
    total = (TP + FP + FN + TN).sum()
    
    accuracy = total_correct / total 
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
    pred_softmax = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_softmax, dim=1)
    
    pred_onehot = F.one_hot(pred_labels, num_classes=num_classes).permute(0, 3, 1, 2).float()
    target_onehot = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2).float()
    
    pred_boundaries = torch.zeros_like(pred_onehot)
    target_boundaries = torch.zeros_like(target_onehot)
    
    for c in range(num_classes):
        pred_edges = torch.abs(sobel(pred_onehot[:, c:c+1]))
        target_edges = torch.abs(sobel(target_onehot[:, c:c+1]))
        
        pred_threshold = pred_edges.mean() + pred_edges.std()
        target_threshold = target_edges.mean() + target_edges.std()
        
        pred_boundaries[:, c:c+1] = (pred_edges > pred_threshold).float()
        target_boundaries[:, c:c+1] = (target_edges > target_threshold).float()
    
    boundary_metrics = {'boundary_precision': 0.0, 'boundary_recall': 0.0, 'boundary_f1': 0.0}
    valid_classes = 0
    
    for c in range(num_classes):
        if target_boundaries[:, c].sum() > 0:
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
    pred_softmax = F.softmax(pred, dim=1)
    pred_labels = torch.argmax(pred_softmax, dim=1)
    separation_scores = []
    
    for b in range(pred_labels.shape[0]):
        batch_scores = []
        for c in range(num_classes):
            pred_mask = (pred_labels[b] == c).cpu().numpy()
            target_mask = (target[b] == c).cpu().numpy()
            
            pred_instances = label(pred_mask)
            target_instances = label(target_mask)
            if target_instances.max() > 0:
                pred_instance_count = len(np.unique(pred_instances)) - 1  # subtract 1 for background
                target_instance_count = len(np.unique(target_instances)) - 1
                
                if target_instance_count == 0:
                    score = 1.0 if pred_instance_count == 0 else 0.0
                else:
                    score = max(0.0, min(1.0, 1.0 - abs(pred_instance_count - target_instance_count) / target_instance_count))
                batch_scores.append(score)
        
        if batch_scores:
            separation_scores.append(np.mean(batch_scores))
    
    return torch.tensor(np.mean(separation_scores) if separation_scores else 0.0).to(pred.device)