import torch
import matplotlib.pyplot as plt
from kornia.filters import sobel
import torch.nn.functional as F
from metrics import (
    calculate_boundary_metrics,
    calculate_confusion_metrics,
    calculate_mean_iou,
    calculate_accuracy,
    calculate_mean_precision,
    calculate_mean_recall,
    calculate_iou_per_class,
    plot_metric_heatmap
)
from config import (
    VISUALIZATION,
    RESULTS_DIR
)
import os

def compute_boundaries(mask, threshold=0.5):
    """
    Compute boundaries from a mask using Sobel filter.
    """
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)
    
    # add batch and channel dimensions if needed
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    edges = torch.abs(sobel(mask))
    boundaries = (edges > threshold).float()
    boundaries = boundaries.squeeze()
    return boundaries

def visualize_batch_with_boundaries(epoch, img_tensor, mask_tensor, outputs, loss, N_CLASSES, index=0, training=True):
    """
    Visualize training progress including boundary detection and post-processing.
    """
    with torch.no_grad():
        outputs_cpu = outputs.detach().cpu()
        img_tensor_cpu = img_tensor.detach().cpu()
        mask_tensor_cpu = mask_tensor.detach().cpu()
        
        pred = outputs_cpu[index]
        pred_classes = torch.argmax(pred, dim=0)

        pred_classes_gpu = torch.argmax(outputs, dim=1)
        results = calculate_confusion_metrics(N_CLASSES, pred_classes_gpu, mask_tensor)
        mean_iou = calculate_mean_iou(results).item()
        accuracy = calculate_accuracy(results).item()
        mean_precision = calculate_mean_precision(results).item()
        mean_recall = calculate_mean_recall(results).item()
        boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, N_CLASSES)
        pred_softmax = F.softmax(outputs, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)

        pred_onehot = (
            F.one_hot(pred_labels, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )
        target_onehot = (
            F.one_hot(mask_tensor, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )
        combined_pred_boundary = torch.zeros_like(pred_labels[0].float())
        combined_target_boundary = torch.zeros_like(mask_tensor[0].float())
        
        for c in range(N_CLASSES):
            pred_edges = torch.abs(sobel(pred_onehot[:, c : c + 1]))
            target_edges = torch.abs(sobel(target_onehot[:, c : c + 1]))

            combined_pred_boundary += (pred_edges[0, 0] > 0.5).float()
            combined_target_boundary += (target_edges[0, 0] > 0.5).float()
        plt.figure(figsize=VISUALIZATION['figure_size'])
        
        # Original image
        plt.subplot(2, 3, 1)
        img_display = img_tensor_cpu[index].permute(1, 2, 0)
        plt.imshow(img_display)
        plt.title("Preprocessed Image")
        plt.axis('off')

        # Ground truth mask
        plt.subplot(2, 3, 2)
        plt.imshow(mask_tensor_cpu[index], cmap='tab10')
        plt.title("Ground Truth")
        plt.axis('off')

        # Prediction mask
        plt.subplot(2, 3, 3)
        plt.imshow(pred_classes, cmap='tab10')
        plt.title(f"Prediction (Epoch {epoch+1})")
        plt.axis('off')
        
        # IoU Heatmap
        plt.subplot(2, 3, 4)
        plt.title("Mean IOU")
        plt.axis("off")
        iou_scores = calculate_iou_per_class(results)
        class_labels = [
            "Year",
            "Date",
            "Latitude",
            "Longitude",
            "Temperature",
            "Background",
        ]
        plot_metric_heatmap(iou_scores, "IoU", class_labels)

        # Ground truth boundaries
        plt.subplot(2, 3, 5)
        plt.imshow(combined_target_boundary.cpu().numpy(), cmap="gray")
        plt.title("Ground Truth Boundaries")
        plt.axis("off")

        # Predicted boundaries
        plt.subplot(2, 3, 6)
        plt.imshow(combined_pred_boundary.cpu().numpy(), cmap="gray")
        plt.title("Predicted Boundaries")
        plt.axis("off")

        # Add metrics in suptitle
        plt.suptitle(
            f'Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f}\n'
            f'Boundary F1: {boundary_metrics["boundary_f1"]:.4f} | '
            f'Mean Precision: {mean_precision:.4f} | '
            f'Mean Recall: {mean_recall:.4f}',
            fontsize=12
        )

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if training:
            save_path = os.path.join(RESULTS_DIR, f'training_epoch_{epoch+1}.png')
        else:
            save_path = os.path.join(RESULTS_DIR, f'validation_epoch_{epoch+1}.png')
        plt.savefig(save_path)
        
        plt.close()
        print(f"Saved visualization to {save_path}")