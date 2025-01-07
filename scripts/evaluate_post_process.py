import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import PirateLogDataset
from post_process import remove_small_regions, separate_tall_regions, erode_regions, create_bounding_boxes
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import os

def count_boxes_per_class(mask, num_classes=6):
    """
    Count the number of distinct regions (boxes) for each class in the mask.
    
    Args:
        mask: numpy array or torch tensor with class labels
        num_classes: number of classes (including background)
    
    Returns:
        dict: Dictionary with class indices as keys and box counts as values
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    box_counts = {}
    for class_idx in range(num_classes - 1):  # Exclude background class
        # Create binary mask for current class
        class_mask = (mask == class_idx).astype(np.uint8)
        
        # Label connected components
        labeled_mask, _ = label(class_mask, return_num=True)
        
        # Count regions with reasonable size
        valid_regions = 0
        for region in regionprops(labeled_mask):
            if region.area > 100:  # Minimum area threshold to filter noise
                valid_regions += 1
                
        box_counts[class_idx] = valid_regions
    
    return box_counts

def post_process_mask(logits):
    """Apply the complete post-processing pipeline to model outputs"""
    pred_classes = torch.argmax(logits, dim=1)
    
    if pred_classes.dim() == 3:
        processed_list = []
        for i in range(pred_classes.shape[0]):
            single = pred_classes[i]
            cleaned = remove_small_regions(single, min_region_size=3000)
            separated = separate_tall_regions(cleaned, expected_row_height=50)
            eroded = erode_regions(separated)
            final = create_bounding_boxes(eroded)
            processed_list.append(final)
        return torch.stack(processed_list)
    else:
        cleaned = remove_small_regions(pred_classes, min_region_size=3000)
        separated = separate_tall_regions(cleaned, expected_row_height=50)
        eroded = erode_regions(separated)
        final = create_bounding_boxes(eroded)
        return final

def extract_row_col_bboxes(final_mask, num_classes=6):
    """
    Extract bounding boxes and organize them by row and column position.
    Returns list of tuples: (row_num, col_num, x1, y1, x2, y2)
    """
    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    regions = []
    
    # Process each class (column)
    for class_id in range(5):  # Excluding background class
        mask = (final_mask == class_id).astype(np.uint8)
        labeled = label(mask)
        for region in regionprops(labeled):
            y_min, x_min, y_max, x_max = region.bbox
            regions.append((y_min, class_id + 1, x_min, y_min, x_max, y_max))
    
    # Sort regions by vertical position
    regions.sort()
    
    # Assign row numbers based on vertical position
    row_col_boxes = []
    current_row = 0
    last_y = -float('inf')
    
    for y_min, col_num, x_min, y_min, x_max, y_max in regions:
        if y_min - last_y > 20:  # Threshold for new row
            current_row += 1
        row_col_boxes.append((current_row, col_num, x_min, y_min, x_max, y_max))
        last_y = y_min
    
    return row_col_boxes

def visualize_comparison(image, gt_mask, pred_mask, index, save_dir):
    """
    Visualize ground truth and prediction side by side with detailed statistics.
    """
    class_names = ["Year", "Date", "Latitude", "Longitude", "Temperature"]
    
    # Calculate statistics
    gt_counts = count_boxes_per_class(gt_mask)
    pred_counts = count_boxes_per_class(pred_mask)
    
    # Get row-column organized boxes for prediction
    pred_row_col_boxes = extract_row_col_bboxes(pred_mask)
    num_rows = max([box[0] for box in pred_row_col_boxes]) if pred_row_col_boxes else 0
    
    # Create statistics text
    stats_text = "Box Counts:\n"
    stats_text += "Class      | GT | Pred | Diff\n"
    stats_text += "-" * 30 + "\n"
    
    for class_idx in range(5):
        gt_count = gt_counts[class_idx]
        pred_count = pred_counts[class_idx]
        diff = pred_count - gt_count
        diff_percent = (diff / gt_count * 100) if gt_count != 0 else 0
        
        stats_text += f"{class_names[class_idx]:<10} | {gt_count:2d} | {pred_count:4d} | {diff:+d} ({diff_percent:+.1f}%)\n"
    
    stats_text += f"\nDetected Rows: {num_rows}"
    
    # Calculate pixel differences
    pixel_diff = (pred_mask != gt_mask).float().sum().item()
    total_pixels = pred_mask.numel()
    pixel_diff_percent = (pixel_diff / total_pixels) * 100
    stats_text += f"\nPixel Differences: {int(pixel_diff):,d} ({pixel_diff_percent:.1f}% of image)"
    
    plt.figure(figsize=(20, 8))
    
    # Original image
    plt.subplot(1, 4, 1)
    img_display = image.permute(1, 2, 0).cpu().numpy()
    plt.imshow(img_display)
    plt.title("Original Image")
    plt.axis('off')
    
    # Ground truth mask
    plt.subplot(1, 4, 2)
    plt.imshow(gt_mask.cpu(), cmap='tab10')
    plt.title("Ground Truth")
    plt.axis('off')
    
    # Prediction mask
    plt.subplot(1, 4, 3)
    plt.imshow(pred_mask.cpu(), cmap='tab10')
    plt.title("Prediction")
    plt.axis('off')
    
    # Statistics text
    plt.subplot(1, 4, 4)
    plt.text(0.1, 0.5, stats_text, fontfamily='monospace', fontsize=10,
             verticalalignment='center', horizontalalignment='left')
    plt.axis('off')
    plt.title("Statistics")
    
    plt.suptitle(f"Image {index:04d} Analysis", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comparison_{index:04d}.png", bbox_inches='tight', dpi=150)
    plt.close()

def plot_box_count_comparison(pred_counts, gt_counts, save_path):
    """
    Create visualization comparing predicted vs ground truth box counts,
    including mean absolute deviation statistics.
    """
    class_names = ["Year", "Date", "Latitude", "Longitude", "Temperature"]
    
    # Prepare data and calculate statistics
    plot_data = []
    total_abs_dev = 0
    n_samples = len(pred_counts)
    class_devs = []
    
    for class_idx in range(5):  # Exclude background
        pred_boxes = [counts[class_idx] for counts in pred_counts]
        gt_boxes = [counts[class_idx] for counts in gt_counts]
        
        # Calculate mean values
        mean_pred = np.mean(pred_boxes)
        mean_gt = np.mean(gt_boxes)
        
        # Calculate mean absolute deviation
        abs_deviations = [abs(p - g) for p, g in zip(pred_boxes, gt_boxes)]
        mean_abs_dev = np.mean(abs_deviations)
        class_devs.append(mean_abs_dev)
        total_abs_dev += sum(abs_deviations)
        
        plot_data.append({
            'Class': class_names[class_idx],
            'Predicted': mean_pred,
            'Ground Truth': mean_gt,
            'MAD': mean_abs_dev
        })
    
    # Calculate overall mean absolute deviation
    overall_mad = total_abs_dev / (n_samples * len(class_names))
    
    df = pd.DataFrame(plot_data)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
    # First subplot: Box counts
    x = np.arange(len(class_names))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df['Predicted'], width, label='Predicted', color='skyblue')
    bars2 = ax1.bar(x + width/2, df['Ground Truth'], width, label='Ground Truth', color='lightgreen')
    
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Average Number of Boxes')
    ax1.set_title('Comparison of Predicted vs Ground Truth Box Counts')
    ax1.set_xticks(x)
    ax1.set_xticklabels(class_names, rotation=45)
    ax1.legend()
    
    # Add value labels on the bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    # Second subplot: Mean Absolute Deviation
    ax2.bar(x, df['MAD'], width, color='salmon', alpha=0.7)
    ax2.axhline(y=overall_mad, color='red', linestyle='--', label=f'Overall MAD: {overall_mad:.2f}')
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Mean Absolute Deviation')
    ax2.set_title('Mean Absolute Deviation per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    
    # Add value labels for MAD bars
    for i, v in enumerate(df['MAD']):
        ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    # Create output directory
    output_dir = "results/evaluation/row50_more_resnet_tversky_weighted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model.load_state_dict(torch.load('results/resnet_tversky_boundary_weighted/best_model.pth'))
    model = model.to(device)
    model.eval()
    
    # Load validation dataset
    val_dataset = PirateLogDataset(
        img_dir="data/processed/val/images",
        mask_dir="data/processed/val/masks",
        target_size=(3200 // 2, 2496 // 2),
        num_classes=6
    )
    
    # Process and visualize each image individually
    print("Processing and visualizing images...")
    all_pred_counts = []
    all_gt_counts = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Processing images"):
            image, gt_mask = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            # Get prediction and post-process
            logits = model(image)['out']
            processed_pred = post_process_mask(logits)
            
            # Count boxes
            pred_counts = count_boxes_per_class(processed_pred[0])
            gt_counts = count_boxes_per_class(gt_mask)
            
            all_pred_counts.append(pred_counts)
            all_gt_counts.append(gt_counts)
            
            # Visualize comparison
            visualize_comparison(
                image[0], 
                gt_mask, 
                processed_pred[0], 
                idx,
                output_dir
            )
    
    # Calculate statistics per class
    class_names = ["Year", "Date", "Latitude", "Longitude", "Temperature"]
    print("\nBox Count Statistics:")
    print("-" * 50)
    
    for class_idx in range(5):  # Exclude background
        pred_boxes = [counts[class_idx] for counts in all_pred_counts]
        gt_boxes = [counts[class_idx] for counts in all_gt_counts]
        
        mean_pred = np.mean(pred_boxes)
        mean_gt = np.mean(gt_boxes)
        diff_percent = ((mean_pred - mean_gt) / mean_gt * 100) if mean_gt != 0 else 0
        
        print(f"\n{class_names[class_idx]}:")
        print(f"  Average predicted boxes: {mean_pred:.2f}")
        print(f"  Average ground truth boxes: {mean_gt:.2f}")
        print(f"  Difference: {diff_percent:+.2f}%")
        print(f"  Per-image counts - Predicted: {pred_boxes}")
        print(f"  Per-image counts - Ground Truth: {gt_boxes}")
    
    # Create final visualization
    plot_box_count_comparison(all_pred_counts, all_gt_counts, f"{output_dir}/box_count_comparison.png")
    print(f"\nVisualizations saved in {output_dir}")

if __name__ == "__main__":
    main()