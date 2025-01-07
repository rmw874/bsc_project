import numpy as np
import cv2
from skimage.measure import label, regionprops
from skimage.filters import gaussian
import torch
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def is_local_minimum(profile, idx):
    """Check if point is local minimum in 5-point neighborhood"""
    return (profile[idx] < profile[idx-1] and 
            profile[idx] < profile[idx+1] and
            profile[idx] < profile[idx-2] and 
            profile[idx] < profile[idx+2])

def calculate_valley_depth(profile, idx):
    """Calculate depth of valley relative to surrounding peaks"""
    left_peak = max(profile[max(0, idx-5):idx])
    right_peak = max(profile[idx+1:min(len(profile), idx+6)])
    return min(left_peak - profile[idx], right_peak - profile[idx])

def find_valleys_in_profile(profile_norm):
    """Find valleys in normalized intensity profile"""
    valleys = []
    min_valley_depth = 0.01
    
    for i in range(2, len(profile_norm)-2):
        if is_local_minimum(profile_norm, i):
            valley_depth = calculate_valley_depth(profile_norm, i)
            if valley_depth > min_valley_depth:
                valleys.append((i, profile_norm[i]))
    
    return valleys

def get_filtered_split_points(valleys, y_min, height, expected_row_height):
    """Get filtered list of split points with minimum separation"""
    # Sort valleys by depth and estimate rows
    valleys.sort(key=lambda x: x[1])
    est_rows = max(2, int(np.ceil(height / expected_row_height)))
    
    # Get initial split points
    split_points = sorted([y_min + v[0] for v in valleys[:est_rows-1]])
    
    # Filter points by minimum distance
    filtered_splits = []
    min_split_distance = expected_row_height * 0.7
    last_split = -float('inf')
    
    for split in split_points:
        if split - last_split >= min_split_distance:
            filtered_splits.append(split)
            last_split = split
            
    return filtered_splits

def separate_tall_regions_watershed(pred_mask, expected_row_height=50):
    """Separate vertically merged regions using watershed segmentation"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = processing_mask.copy()
        
    result_mask = processing_mask.copy()
    
    # Process each class separately
    for class_idx in range(5):
        # Get binary mask for current class
        class_mask = (processing_mask == class_idx).astype(np.uint8)
        if not np.any(class_mask):
            continue
            
        # Label initial connected components
        labeled_mask, _ = label(class_mask, return_num=True)
        
        # Process each component
        for region in regionprops(labeled_mask):
            y_min, x_min, y_max, x_max = region.bbox
            height = y_max - y_min
            width = x_max - x_min
            
            # Only process regions that are wider than they are tall
            if width < height * 0.5:
                continue
                
            # Process if taller than expected
            if height > expected_row_height * 1.15:
                # Extract region image
                region_img = region.image
                
                # Calculate distance transform
                distance = ndi.distance_transform_edt(region_img)
                
                # Find local maxima with minimum distance based on expected height
                min_distance = int(expected_row_height * 0.7)
                coords = peak_local_max(distance, min_distance=min_distance)
                
                # Create markers for watershed
                markers = np.zeros_like(region_img, dtype=int)
                for i, (y, x) in enumerate(coords, start=1):
                    markers[y, x] = i
                    
                # Apply watershed
                labels = watershed(-distance, markers, mask=region_img)
                
                # Create separations between watershed regions
                boundaries = ndi.find_objects(labels)
                for i in range(len(boundaries)-1):
                    y_split = (boundaries[i][0].stop + boundaries[i+1][0].start) // 2
                    # Add separation line to result mask
                    y_global = y_min + y_split
                    result_mask[y_global:y_global+3, x_min:x_max] = 5  # Background class
    if isinstance(pred_mask, torch.Tensor): 
        return torch.from_numpy(result_mask) 
    else: 
        return result_mask

def separate_tall_regions(pred_mask, expected_row_height=30, min_row_separation=3):
    """Separate vertically merged regions by analyzing height profiles"""
    # Convert to numpy if needed
    processing_mask = pred_mask.cpu().numpy() if isinstance(pred_mask, torch.Tensor) else pred_mask.copy()
    result_mask = processing_mask.copy()
    
    # Process each class
    for class_idx in range(5):
        class_mask = processing_mask == class_idx
        if not np.any(class_mask):
            continue
            
        labeled_mask, _ = label(class_mask, return_num=True)
        
        # Process each region
        for region in regionprops(labeled_mask):
            y_min, x_min, y_max, x_max = region.bbox
            height = y_max - y_min
            width = x_max - x_min
            
            # Skip narrow regions
            if width < height * 0.5:
                continue
                
            # Process tall regions
            if height > expected_row_height * 1.15:
                # Get vertical profile
                profile = np.sum(region.image, axis=1)
                sigma = max(1, height / expected_row_height * 0.5)
                profile_smooth = gaussian(profile, sigma=sigma)
                
                # Normalize profile
                profile_norm = (profile_smooth - profile_smooth.min()) / (profile_smooth.max() - profile_smooth.min())
                
                # Find valleys and get split points
                valleys = find_valleys_in_profile(profile_norm)
                if valleys:
                    split_points = get_filtered_split_points(valleys, y_min, height, expected_row_height)
                    
                    # Create separations
                    for y in split_points:
                        y_start = max(0, y - min_row_separation)
                        y_end = min(processing_mask.shape[0], y + min_row_separation + 1)
                        result_mask[y_start:y_end, x_min:x_max] = 5
    if isinstance(pred_mask, torch.Tensor): 
        return torch.from_numpy(result_mask) 
    else: 
        return result_mask

def remove_small_regions(pred_mask, min_region_size=3000, background_class=5):
    """Remove small predicted regions that are likely noise"""
    if isinstance(pred_mask, torch.Tensor):
        processing_mask = pred_mask.cpu().numpy()
    else:
        processing_mask = pred_mask.copy()
    
    for class_idx in range(5):
        class_mask = processing_mask == class_idx
        if class_idx == 0 or not np.any(class_mask):
            continue

            
        labeled_mask, num_features = label(class_mask, return_num=True)
        for region_idx in range(1, num_features + 1):

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
    
def post_process_mask(logits, min_region_size=1500, expected_row_height=40):
    """Apply the complete post-processing pipeline to model outputs"""
    pred_classes = torch.argmax(logits, dim=1)  
    
    if pred_classes.dim() == 3:
        processed_list = []
        for i in range(pred_classes.shape[0]):
            single = pred_classes[i]
            cleaned = remove_small_regions(single, min_region_size=min_region_size)
            separated = separate_tall_regions(cleaned, expected_row_height=expected_row_height)
            eroded = erode_regions(separated)
            final = create_bounding_boxes(eroded)
            processed_list.append(final)
        return torch.stack(processed_list)
    else:
        cleaned = remove_small_regions(pred_classes, min_region_size=min_region_size)
        separated = separate_tall_regions(cleaned, expected_row_height=expected_row_height)
        eroded = erode_regions(separated)
        final = create_bounding_boxes(eroded)
        return final

def extract_row_col_bboxes(final_mask, num_classes=6, row_threshold=40):
    """
    Extract bounding boxes and organize them by row and column position using y-centers.
    
    Args:
        final_mask: Segmentation mask
        num_classes: Number of classes in the segmentation
        row_threshold: Minimum vertical distance to consider as a new row
        
    Returns:
        list of tuples: (row_num, col_num, x1, y1, x2, y2)
    """
    from skimage.measure import label, regionprops

    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    # Store regions with their positions and centers
    regions = []
    
    # Process each class (column)
    for class_id in range(5):  # Excluding background class
        mask = (final_mask == class_id).astype(np.uint8)
        labeled = label(mask)
        
        for region in regionprops(labeled):
            y_min, x_min, y_max, x_max = region.bbox
            y_center = (y_max + y_min) / 2  # Calculate y-center
            
            # Store y_center for sorting and row assignment
            regions.append((y_center, class_id, x_min, y_min, x_max, y_max))
            
    # Sort regions by y-center position
    regions.sort()
    
    # Assign row numbers based on y-center clustering
    row_col_boxes = []
    current_row = 0
    if regions:
        # Initialize with first region's y-center
        current_y_center = regions[0][0]
        
        for y_center, col_num, x_min, y_min, x_max, y_max in regions:
            # If there's a significant gap between y-centers, increment row number
            if abs(y_center - current_y_center) > row_threshold:
                current_row += 1
                current_y_center = y_center
                
            row_col_boxes.append((current_row, col_num, x_min, y_min, x_max, y_max))
    
    # Sort final results by row then column for consistent ordering
    row_col_boxes.sort(key=lambda x: (x[0], x[1]))
    
    return row_col_boxes