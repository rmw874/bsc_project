import os
import random
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import preprocess
from post_process import post_process_mask

def load_test_images(bottom_dir, top_dir, num_bottom=5, num_top=5):
    """Load test images from bottom and top page directories"""
    bottom_files = [f for f in os.listdir(bottom_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    top_files = [f for f in os.listdir(top_dir) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    random.shuffle(bottom_files)
    random.shuffle(top_files)

    bottom_chosen = [os.path.join(bottom_dir, f) for f in bottom_files[:num_bottom]]
    top_chosen = [os.path.join(top_dir, f) for f in top_files[:num_top]]

    return bottom_chosen + top_chosen

def extract_row_col_bboxes(final_mask, num_classes=6, row_threshold=30):
    """Extract bounding boxes with row and column information"""
    from skimage.measure import label, regionprops

    if isinstance(final_mask, torch.Tensor):
        final_mask = final_mask.cpu().numpy()

    regions = []
    
    for class_id in range(5):  # Excluding background class
        mask = (final_mask == class_id).astype(np.uint8)
        labeled = label(mask)
        
        for region in regionprops(labeled):
            y_min, x_min, y_max, x_max = region.bbox
            y_center = (y_max + y_min) / 2
            regions.append((y_center, class_id + 1, x_min, y_min, x_max, y_max))
    
    regions.sort()
    
    row_col_boxes = []
    current_row = 0
    if regions:
        current_y_center = regions[0][0]
        
        for y_center, col_num, x_min, y_min, x_max, y_max in regions:
            if abs(y_center - current_y_center) > row_threshold:
                current_row += 1
                current_y_center = y_center
                
            row_col_boxes.append((current_row, col_num, x_min, y_min, x_max, y_max))
    
    row_col_boxes.sort(key=lambda x: (x[0], x[1]))
    return row_col_boxes

def scale_bbox(bbox, original_size, resized_size):
    """Scale bounding box coordinates from resized to original image dimensions"""
    row_num, col_num, x1, y1, x2, y2 = bbox
    w_orig, h_orig = original_size
    w_resized, h_resized = resized_size
    
    # Scale coordinates
    x1_orig = int(x1 * (w_orig / w_resized))
    y1_orig = int(y1 * (h_orig / h_resized))
    x2_orig = int(x2 * (w_orig / w_resized))
    y2_orig = int(y2 * (h_orig / h_resized))
    
    return (row_num, col_num, x1_orig, y1_orig, x2_orig, y2_orig)

def process_image_for_ocr(image, processor, min_size=32):
    """Preprocess image for OCR with size checking and padding"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Get original dimensions
    width, height = image.size
    
    # Add padding if image is too small
    if width < min_size or height < min_size:
        new_width = max(width, min_size)
        new_height = max(height, min_size)
        padded_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        padded_image.paste(image, (0, 0))
        image = padded_image
    
    # Process image for TrOCR
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def main():
    random.seed(42)
    
    # Configure paths
    bottom_dir = "data/processed/Mathiesen-single-pages/temp"
    top_dir = "data/processed/Mathiesen-single-pages/temp"
    out_dir = "results/ocr/row_col_crops"
    os.makedirs(out_dir, exist_ok=True)

    # Load and prepare segmentation model
    model_img = deeplabv3_resnet50(pretrained=True)
    model_img.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model_img.load_state_dict(torch.load('results/resnet_tversky_vanilla_lower_batch/best_model.pth'))
    model_img.cuda()
    model_img.eval()

    # Load OCR model and processor - using large model for better accuracy
    model_ocr = VisionEncoderDecoderModel.from_pretrained("results/trocr_finetuned_1250/final_model")
    processor = TrOCRProcessor.from_pretrained("results/trocr_finetuned_1250/final_model")
    model_ocr.to('cuda')  # Move OCR model to GPU

    # Process test images
    test_paths = load_test_images(bottom_dir, top_dir, 2, 0)
    
    for img_path in test_paths:
        print(f"\nProcessing: {img_path}")
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        
        # Load and preprocess image
        original_pil = Image.open(img_path).convert('RGB')
        bgr_img = cv2.imread(img_path)
        if bgr_img is None:
            continue
        
        # Get original and resized dimensions
        h_orig, w_orig = bgr_img.shape[:2]
        target_size = (1600, 1248)
        preprocessed = preprocess(bgr_img, target_size)
        h_resized, w_resized = preprocessed.shape[:2]

        # Prepare tensor for model
        img_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).cuda()

        # Run inference and post-processing
        with torch.no_grad():
            logits = model_img(img_tensor)['out']
        final_mask = post_process_mask(logits)
        
        if final_mask.dim() == 3:
            final_mask = final_mask[0]

        # Extract bounding boxes and scale to original size
        bboxes = extract_row_col_bboxes(final_mask)
        scaled_bboxes = [
            scale_bbox(bbox, (w_orig, h_orig), (w_resized, h_resized))
            for bbox in bboxes
        ]
        
        outputs = []
        
        # Process each region with OCR
        for row_num, col_num, x1, y1, x2, y2 in scaled_bboxes:
            # Crop from original image using scaled coordinates
            crop = original_pil.crop((x1, y1, x2, y2))
            
            try:
                # Process image for OCR
                pixel_values = process_image_for_ocr(crop, processor)
                pixel_values = pixel_values.to(model_ocr.device)
                
                # Generate OCR prediction with improved parameters
                generated_ids = model_ocr.generate(
                    pixel_values,
                    max_length=128,
                    num_beams=4,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.0,
                    length_penalty=1.0,
                    no_repeat_ngram_size=3
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                outputs.append((row_num, col_num, generated_text))
                
                # Save crop for debugging
                crop_path = os.path.join(out_dir, f"{base_filename}_row_{row_num}_col_{col_num}.png")
                crop.save(crop_path)
                
            except Exception as e:
                print(f"Error processing region (row {row_num}, col {col_num}): {str(e)}")
                continue
        
        # Sort and display results
        outputs.sort()
        print("\nExtracted text by row and column:")
        current_row = None
        for row_num, col_num, text in outputs:
            if current_row != row_num:
                print(f"\nRow {row_num}:")
                current_row = row_num
            print(f"  Column {col_num}: {text}")

if __name__ == "__main__":
    main()