import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.filters import gaussian
import torch

def separate_tall_regions(pred_mask, expected_row_height=72, min_row_separation=3):
    """Separate vertically merged regions by analyzing height profiles"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = pred_mask.copy()
        
    result_mask = processing_mask.copy()
    
    # Process each class separately
    for class_idx in range(5):
        # Get binary mask for current class
        class_mask = processing_mask == class_idx
        if not np.any(class_mask):
            continue
            
        # Label connected components
        labeled_mask, _ = label(class_mask, return_num=True)
        
        # Process each component
        for region in regionprops(labeled_mask):
            y_min, x_min, y_max, x_max = region.bbox
            height = y_max - y_min
            width = x_max - x_min
            
            # Only process regions that are wider than they are tall (valid cells)
            if width < height * 0.5:  # Skip very narrow regions
                continue
                
            # Process if taller than expected
            if height > expected_row_height * 1.15:  # Slightly more sensitive
                # Get vertical profile of the region
                profile = np.sum(region.image, axis=1)
                # Adaptive smoothing based on height
                sigma = max(1, height / expected_row_height * 0.5)
                profile_smooth = gaussian(profile, sigma=sigma)
                
                # Normalize profile
                profile_norm = (profile_smooth - profile_smooth.min()) / (profile_smooth.max() - profile_smooth.min())
                
                # Find local minima
                valleys = []
                min_valley_depth = 0.01  # Valley must be 40% lower than peaks
                for i in range(2, len(profile_norm)-2):
                    # Look at 5-point neighborhood
                    if (profile_norm[i] < profile_norm[i-1] and 
                        profile_norm[i] < profile_norm[i+1] and
                        profile_norm[i] < profile_norm[i-2] and 
                        profile_norm[i] < profile_norm[i+2]):
                        # Check valley depth relative to surrounding peaks
                        left_peak = max(profile_norm[max(0, i-5):i])
                        right_peak = max(profile_norm[i+1:min(len(profile_norm), i+6)])
                        valley_depth = min(left_peak - profile_norm[i], right_peak - profile_norm[i])
                        
                        if valley_depth > min_valley_depth:
                            valleys.append((i, profile_norm[i]))
                
                if valleys:
                    # Sort valleys by depth
                    valleys.sort(key=lambda x: x[1])
                    
                    # Estimate number of rows
                    est_rows = max(2, int(np.ceil(height / expected_row_height)))
                    # print(f"Class {class_idx} - Region height: {height}, width: {width}, estimated rows: {est_rows}, found {len(valleys)} valleys")
                    
                    # Take the deepest valleys
                    split_points = sorted([y_min + v[0] for v in valleys[:est_rows-1]])
                    
                    # Enforce minimum distance between split points
                    filtered_splits = []
                    min_split_distance = expected_row_height * 0.7
                    
                    last_split = -float('inf')
                    for split in split_points:
                        if split - last_split >= min_split_distance:
                            filtered_splits.append(split)
                            last_split = split
                    
                    # Create separations
                    for y in filtered_splits:
                        y_start = max(0, y - min_row_separation)
                        y_end = min(processing_mask.shape[0], y + min_row_separation + 1)
                        result_mask[y_start:y_end, x_min:x_max] = 5
    
    if isinstance(pred_mask, torch.Tensor):
        return torch.from_numpy(result_mask)  
    else:
        return result_mask

def remove_small_regions(pred_mask, min_region_size=1000, background_class=5):
    """Remove small predicted regions that are likely noise"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = pred_mask.copy()
    
    for class_idx in range(5):
        class_mask = processing_mask == class_idx
        if not np.any(class_mask):
            continue
            
        labeled_mask, num_features = label(class_mask, return_num=True)
        for region_idx in range(1, num_features + 1):
            if region_idx == 1:
                min_region_size = 500 # since years are usually way smaller
            region = labeled_mask == region_idx
            region_size = np.sum(region)
            if region_size < min_region_size:
                processing_mask[region] = background_class
                # print(f"Removed region of class {class_idx} with size {region_size}")
    
    return torch.from_numpy(processing_mask) if isinstance(pred_mask, torch.Tensor) else processing_mask

def erode_regions(pred_mask, kernel_size=3, background_class=5):
    """Erode each class region to create separation between touching cells"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = pred_mask.copy()
    
    result_mask = np.full_like(processing_mask, background_class)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Process each class separately (excluding background)
    for class_idx in range(5):
        # Get binary mask for current class
        class_mask = (processing_mask == class_idx).astype(np.uint8)
        if not np.any(class_mask):
            continue
            
        # Apply erosion
        eroded_mask = cv2.erode(class_mask, kernel, iterations=1)
        
        # Update result mask with eroded regions
        result_mask[eroded_mask > 0] = class_idx

    return torch.from_numpy(result_mask) if isinstance(pred_mask, torch.Tensor) else result_mask


def create_bounding_boxes(pred_mask, expected_row_height=72//2):
    """Create clean rectangular bounding boxes for each region"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = pred_mask.copy()
    
    result_mask = processing_mask.copy()
    
    # Process each class separately
    for class_idx in range(5):
        # Get binary mask for current class
        class_mask = processing_mask == class_idx
        if not np.any(class_mask):
            continue
            
        # Label connected components
        labeled_mask, num_features = label(class_mask, return_num=True)
        
        # Process each component
        for region in regionprops(labeled_mask):
            if region.area < expected_row_height * expected_row_height:
                continue
                
            # Get bounding box coordinates
            y_min, x_min, y_max, x_max = region.bbox
            
            # Create rectangular region
            result_mask[y_min:y_max, x_min:x_max] = class_idx
            
            # print(f"Created box for class {class_idx} with height {y_max-y_min}")
    
    return torch.from_numpy(result_mask) if isinstance(pred_mask, torch.Tensor) else result_mask

def post_process_predictions(pred_logits, min_region_size=1000, expected_row_height=50):
    """Apply full post-processing pipeline"""
    # Get class predictions
    if pred_logits.dim() == 4:  # (B, C, H, W)
        pred_classes = torch.argmax(pred_logits, dim=1)  # (B, H, W)
    else:  # (C, H, W)
        pred_classes = torch.argmax(pred_logits, dim=0)  # (H, W)
    
    # Process each image in batch
    if pred_classes.dim() == 3:
        processed = []
        for i in range(pred_classes.shape[0]):
            # Remove small regions first
            cleaned = remove_small_regions(pred_classes[i], min_region_size)
            
            # Separate tall regions
            separated = separate_tall_regions(cleaned, expected_row_height)
            
            # Add erosion step before creating bounding boxes
            eroded = erode_regions(separated, kernel_size=3)
            
            # Create bounding boxes
            processed_mask = create_bounding_boxes(eroded, expected_row_height)
            processed.append(processed_mask)
        
        return torch.stack(processed)
    else:
        # Single image case
        cleaned = remove_small_regions(pred_classes, min_region_size)
        separated = separate_tall_regions(cleaned, expected_row_height)
        eroded = erode_regions(separated, kernel_size=3)
        bounding_boxes = create_bounding_boxes(eroded, expected_row_height)
        return bounding_boxes