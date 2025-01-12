import torch
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from dataset import PirateLogDataset
from post_process import remove_small_regions, separate_tall_regions, erode_regions, create_bounding_boxes, post_process_mask
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import os

def count_boxes_per_class(mask, num_classes=6):
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    box_counts = {}
    for class_idx in range(num_classes - 1):  # exclude background class
        class_mask = (mask == class_idx).astype(np.uint8)
        
        labeled_mask, _ = label(class_mask, return_num=True)
        
        valid_regions = 0
        for region in regionprops(labeled_mask):
            if region.area > 100: 
                valid_regions += 1
                
        box_counts[class_idx] = valid_regions
    
    return box_counts

def extract_row_col_bboxes(final_mask, num_classes=6):
    """
    Extract bounding boxes and organize them by row and column position.
    """
    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    regions = []
    for class_id in range(5):  
        mask = (final_mask == class_id).astype(np.uint8)
        labeled = label(mask)
        for region in regionprops(labeled):
            y_min, x_min, y_max, x_max = region.bbox
            regions.append((y_min, class_id + 1, x_min, y_min, x_max, y_max))
    
    # sort regions by vertical position
    regions.sort()
    
    # assign row numbers based on vertical position
    row_col_boxes = []
    current_row = 0
    last_y = -float('inf')
    
    for y_min, col_num, x_min, y_min, x_max, y_max in regions:
        if y_min - last_y > 20:  # threshold for new row
            current_row += 1
        row_col_boxes.append((current_row, col_num, x_min, y_min, x_max, y_max))
        last_y = y_min
    
    return row_col_boxes

def visualize_comparison(image, gt_mask, pred_mask, index, save_dir):
    class_names = ["Year", "Date", "Latitude", "Longitude", "Temperature"]
    
    gt_counts = count_boxes_per_class(gt_mask)
    pred_counts = count_boxes_per_class(pred_mask)
    
    pred_row_col_boxes = extract_row_col_bboxes(pred_mask)
    num_rows = max([box[0] for box in pred_row_col_boxes]) if pred_row_col_boxes else 0
    
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
    
    plot_data = []
    total_abs_dev = 0
    n_samples = len(pred_counts)
    class_devs = []
    
    for class_idx in range(5):
        pred_boxes = [counts[class_idx] for counts in pred_counts]
        gt_boxes = [counts[class_idx] for counts in gt_counts]
        
        mean_pred = np.mean(pred_boxes)
        mean_gt = np.mean(gt_boxes)
        
        # mean absolute deviation
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
    
    overall_mad = total_abs_dev / (n_samples * len(class_names))
    
    df = pd.DataFrame(plot_data)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[2, 1])
    
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
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom')
    
    ax2.bar(x, df['MAD'], width, color='salmon', alpha=0.7)
    ax2.axhline(y=overall_mad, color='red', linestyle='--', label=f'Overall MAD: {overall_mad:.2f}')
    
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Mean Absolute Deviation')
    ax2.set_title('Mean Absolute Deviation per Class')
    ax2.set_xticks(x)
    ax2.set_xticklabels(class_names, rotation=45)
    ax2.legend()
    
    for i, v in enumerate(df['MAD']):
        ax2.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

def main():
    output_dir = "results/evaluation/post_process_250epoch_1000_40"
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = deeplabv3_resnet50(pretrained=True)
    model.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model.load_state_dict(torch.load('results/DeepLab3_ResNet50_Tversky_250/best_model.pth'))

    model = model.to(device)
    model.eval()
    
    val_dataset = PirateLogDataset(
        img_dir="data/processed/val/images",
        mask_dir="data/processed/val/masks",
        target_size=(3200 // 2, 2496 // 2),
        num_classes=6
    )
    
    print("Processing and visualizing images...")
    all_pred_counts = []
    all_gt_counts = []
    
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Processing images"):
            image, gt_mask = val_dataset[idx]
            image = image.unsqueeze(0).to(device)
            
            logits = model(image)['out']
            processed_pred = post_process_mask(logits, 1000, 40)
            
            pred_counts = count_boxes_per_class(processed_pred[0])
            gt_counts = count_boxes_per_class(gt_mask)
            
            all_pred_counts.append(pred_counts)
            all_gt_counts.append(gt_counts)
            
            visualize_comparison(
                image[0], 
                gt_mask, 
                processed_pred[0], 
                idx,
                output_dir
            )
    
    class_names = ["Year", "Date", "Latitude", "Longitude", "Temperature"]
    print("\nBox Count Statistics:")
    print("-" * 50)
    
    for class_idx in range(5):
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
    
    plot_box_count_comparison(all_pred_counts, all_gt_counts, f"{output_dir}/box_count_comparison.png")
    print(f"\nVisualizations saved in {output_dir}")

if __name__ == "__main__":
    main()