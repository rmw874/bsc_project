import torch
import numpy as np
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
from post_process import post_process_predictions
from config import (
    LEARNING_RATE,
    BATCH_SIZE,
    RUN_NAME,
    VISUALIZATION,
    RESULTS_DIR,
    POST_PROCESSING
)
import os

def compute_boundaries(mask, threshold=0.5):
    """
    Compute boundaries from a mask using Sobel filter.
    
    Args:
        mask: Input mask tensor or numpy array
        threshold: Threshold for edge detection (default: 0.5)
    
    Returns:
        Tensor of detected boundaries
    """
    # Convert to tensor if needed
    if not isinstance(mask, torch.Tensor):
        mask = torch.from_numpy(mask)
    
    # Add batch and channel dimensions if needed
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.dim() == 3:
        mask = mask.unsqueeze(1)
        
    # Convert to float
    mask = mask.float()
    
    # Apply Sobel filter
    edges = torch.abs(sobel(mask))
    
    # Threshold to get boundaries
    boundaries = (edges > threshold).float()
    
    # Remove extra dimensions
    boundaries = boundaries.squeeze()
    
    return boundaries

def visualize_batch_with_boundaries(epoch, img_tensor, mask_tensor, outputs, loss, N_CLASSES, index=0, include_post_processing=True, training=True):
    """
    Visualize training progress including boundary detection and post-processing.
    
    Args:
        epoch: Current epoch number
        img_tensor: Input image tensor [B, C, H, W]
        mask_tensor: Ground truth mask tensor [B, H, W]
        outputs: Model predictions [B, C, H, W]
        loss: Current loss value
        N_CLASSES: Number of classes
        index: Which batch item to visualize
        include_post_processing: Whether to show post-processed results
    """
    with torch.no_grad():
        # Get device from input tensors
        device = outputs.device
        
        # Move tensors to CPU for visualization
        outputs_cpu = outputs.detach().cpu()
        img_tensor_cpu = img_tensor.detach().cpu()
        mask_tensor_cpu = mask_tensor.detach().cpu()
        
        # Get predictions
        pred = outputs_cpu[index]
        pred_classes = torch.argmax(pred, dim=0)

        # Calculate metrics on GPU before moving to CPU
        pred_classes_gpu = torch.argmax(outputs, dim=1)
        results = calculate_confusion_metrics(N_CLASSES, pred_classes_gpu, mask_tensor)
        mean_iou = calculate_mean_iou(results).item()
        accuracy = calculate_accuracy(results).item()
        mean_precision = calculate_mean_precision(results).item()
        mean_recall = calculate_mean_recall(results).item()
        

        # Calculate boundary metrics
        boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, N_CLASSES)

        # Generate boundary maps
        pred_softmax = F.softmax(outputs, dim=1)
        pred_labels = torch.argmax(pred_softmax, dim=1)

        # Convert to one-hot
        pred_onehot = (
            F.one_hot(pred_labels, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )
        target_onehot = (
            F.one_hot(mask_tensor, num_classes=N_CLASSES).permute(0, 3, 1, 2).float()
        )

        # Compute boundaries for visualization
        combined_pred_boundary = torch.zeros_like(pred_labels[0].float())
        combined_target_boundary = torch.zeros_like(mask_tensor[0].float())
        
        for c in range(N_CLASSES):
            pred_edges = torch.abs(sobel(pred_onehot[:, c : c + 1]))
            target_edges = torch.abs(sobel(target_onehot[:, c : c + 1]))

            combined_pred_boundary += (pred_edges[0, 0] > 0.5).float()
            combined_target_boundary += (target_edges[0, 0] > 0.5).float()
        
        # Create the visualization plot
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

        # Add metrics as suptitle
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

# def visualize_batch_with_boundaries(
#     epoch, img_tensor, mask_tensor, outputs, loss, N_CLASSES, index=0, include_post_processing=True
# ):
#     """
#     Visualize training progress including boundary detection and post-processing

#     Args:
#         epoch: Current training epoch
#         img_tensor: Input image tensor [B, C, H, W]
#         mask_tensor: Ground truth mask tensor [B, H, W]
#         outputs: Model predictions [B, C, H, W]
#         loss: Current loss value
#         N_CLASSES: Number of classes
#         index: Which batch item to visualize
#         include_post_processing: Whether to show post-processed results
#     """
#     with torch.no_grad():
#         # Move tensors to CPU for processing
#         outputs_cpu = outputs.cpu()
#         img_tensor = img_tensor.cpu()
#         mask_tensor = mask_tensor.cpu()
        
#         # Get predictions
#         pred = outputs_cpu[index]
#         pred_classes = torch.argmax(pred, dim=0)

#         # Get post-processed prediction if requested
#         if include_post_processing:
#             post_processed = post_process_predictions(
#                 outputs_cpu[index:index+1],
#                 min_region_size=POST_PROCESSING['min_region_size'],
#                 expected_row_height=POST_PROCESSING['expected_row_height']
#             )
#             post_processed = post_processed[0] if post_processed.dim() > 2 else post_processed

#         img_resized = img_tensor[index].permute(1, 2, 0)
#         mask_resized = mask_tensor[index]

#         # Calculate metrics for original prediction
#         results = calculate_confusion_metrics(N_CLASSES, outputs, mask_tensor)
#         mean_iou = calculate_mean_iou(results).item()
#         accuracy = calculate_accuracy(results).item()
#         mean_precision = calculate_mean_precision(results).item()
#         mean_recall = calculate_mean_recall(results).item()

#         # Calculate boundary metrics
#         boundary_metrics = calculate_boundary_metrics(outputs, mask_tensor, N_CLASSES)
        
#         # Set up the figure - make it larger if including post-processing
#         n_cols = 4 if include_post_processing else 3
#         plt.figure(figsize=VISUALIZATION['figure_size'])

#         # Original image
#         plt.subplot(2, n_cols, 1)
#         plt.imshow(img_resized)
#         plt.title("Preprocessed Image")
#         plt.axis('off')

#         # Ground truth mask
#         plt.subplot(2, n_cols, 2)
#         plt.imshow(mask_resized, cmap='tab10')
#         plt.title("Ground Truth")
#         plt.axis('off')

#         # Raw prediction mask
#         plt.subplot(2, n_cols, 3)
#         plt.imshow(pred_classes, cmap='tab10')
#         plt.title(f"Raw Prediction (Epoch {epoch+1})")
#         plt.axis('off')

#         # Post-processed prediction
#         if include_post_processing:
#             plt.subplot(2, n_cols, 4)
#             plt.imshow(post_processed, cmap='tab10')
#             plt.title("Post-processed")
#             plt.axis('off')

#         # Boundary visualization and metrics in bottom row
#         n_bottom = n_cols if include_post_processing else 3
        
#         # Predicted boundaries
#         plt.subplot(2, n_bottom, n_bottom + 1)
#         pred_boundaries = compute_boundaries(pred_classes)
#         plt.imshow(pred_boundaries, cmap='gray')
#         plt.title("Predicted Boundaries")
#         plt.axis('off')

#         # Ground truth boundaries
#         plt.subplot(2, n_bottom, n_bottom + 2)
#         gt_boundaries = compute_boundaries(mask_resized)
#         plt.imshow(gt_boundaries, cmap='gray')
#         plt.title("Ground Truth Boundaries")
#         plt.axis('off')

#         # IoU Heatmap
#         plt.subplot(2, n_bottom, n_bottom + 3)
#         iou_scores = calculate_iou_per_class(results)
#         plot_metric_heatmap(iou_scores, "IoU", VISUALIZATION['class_labels'])

#         if include_post_processing:
#             # Show difference map
#             plt.subplot(2, n_bottom, n_bottom + 4)
#             diff = (post_processed != pred_classes).float()
#             plt.imshow(diff, cmap='hot')
#             plt.title("Post-processing Changes")
#             plt.axis('off')

#         # Add metrics as suptitle
#         plt.suptitle(
#             f'Loss: {loss:.4f} | Mean IoU: {mean_iou:.4f} | Accuracy: {accuracy:.4f}\n'
#             f'Boundary F1: {boundary_metrics["boundary_f1"]:.4f} | '
#             f'Boundary Precision: {boundary_metrics["boundary_precision"]:.4f} | '
#             f'Boundary Recall: {boundary_metrics["boundary_recall"]:.4f}\n'
#             f'Mean Precision: {mean_precision:.4f} | '
#             f'Mean Recall: {mean_recall:.4f}',
#             fontsize=16,
#         )

#         plt.tight_layout(rect=[0, 0, 1, 0.95])
#         plt.savefig(f"{RESULTS_DIR}/validation_epoch_{epoch+1}_lr_{LEARNING_RATE}_bs_{BATCH_SIZE}.png")
#         plt.close()
#         print("Saving figure...")

