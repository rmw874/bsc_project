import argparse
import os
import random
import pandas as pd
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from dataset import preprocess
from post_process import post_process_mask, extract_row_col_bboxes

def find_matching_pages(data_dir):
    """
    Find matching top and bottom page pairs in the directory structure.
    Handles both single directory with -t/-b suffixes and top/bottom subdirectories.
    
    Args:
        data_dir: Root directory containing either subdirectories or mixed files
        
    Returns:
        list: List of tuples (top_path, bottom_path)
    """
    top_dir = os.path.join(data_dir, "top")
    bottom_dir = os.path.join(data_dir, "bottom")
    
    if os.path.exists(top_dir) and os.path.exists(bottom_dir):
        top_files = sorted([f for f in os.listdir(top_dir) 
                          if f.lower().endswith('.jpg')])
        bottom_files = sorted([f for f in os.listdir(bottom_dir)
                             if f.lower().endswith('.jpg')])
        
        valid_pairs = []
        top_dict = {os.path.splitext(f)[0].replace('-t', ''): f for f in top_files}
        bottom_dict = {os.path.splitext(f)[0].replace('-b', ''): f for f in bottom_files}
        
        common_bases = set(top_dict.keys()) & set(bottom_dict.keys())
        for base in sorted(common_bases):
            top_path = os.path.join(top_dir, top_dict[base])
            bottom_path = os.path.join(bottom_dir, bottom_dict[base])
            valid_pairs.append((top_path, bottom_path))
            
        if valid_pairs:
            print(f"Found {len(valid_pairs)} pairs in top/bottom directories")
            return valid_pairs
    
    files = [f for f in os.listdir(data_dir) 
             if f.lower().endswith(('.jpg'))]
    
    page_pairs = {}
    for f in files:
        base_name = f.replace('-t.', '.').replace('-b.', '.')
        base_name = os.path.splitext(base_name)[0]
        if base_name not in page_pairs:
            page_pairs[base_name] = {'top': None, 'bottom': None}
            
        if '-t.' in f:
            page_pairs[base_name]['top'] = f
        elif '-b.' in f:
            page_pairs[base_name]['bottom'] = f
    
    valid_pairs = []
    for base_name, pair in page_pairs.items():
        if pair['top'] and pair['bottom']:
            top_path = os.path.join(data_dir, pair['top'])
            bottom_path = os.path.join(data_dir, pair['bottom'])
            valid_pairs.append((top_path, bottom_path))
    
    print(f"Found {len(valid_pairs)} pairs with -t/-b suffixes")
    return valid_pairs

def clean_year_column(df):
    """
    Clean the Year column to contain only numbers.
    
    Args:
        df: pandas DataFrame with a 'Year' column
    
    Returns:
        DataFrame with cleaned Year column
    """
    if 'Year' not in df.columns:
        return df
        
    def extract_year(value):
        value = str(value)
        digits = ''.join(filter(str.isdigit, value))

        if len(digits) == 4:
            return digits
        elif len(digits) > 4:
            return digits[:4]
        else:
            return pd.NA
    
    df['Year'] = df['Year'].apply(extract_year)    
    df['Year'] = df['Year'].replace('', pd.NA).ffill()
    
    return df

def merge_page_dataframes(top_df, bottom_df):
    """
    Merge top and bottom page DataFrames
    """
    if top_df.empty and bottom_df.empty:
        return pd.DataFrame()
    
    merged_df = pd.concat([top_df, bottom_df], ignore_index=True)
    
    if not merged_df.empty:
        for column in merged_df.columns:
            if not merged_df[column].empty and merged_df[column].notna().any():
                merged_df[column] = merged_df[column].replace({r'^\s*\.\s*$': None}, regex=True)
    
    merged_df = clean_year_column(merged_df)
    
    if 'Date' in merged_df.columns:
        merged_df['Date'] = merged_df['Date'].ffill()
    
    return merged_df

def process_single_page(img_path, model_img, model_ocr, processor, out_dir):
    """
    Process a single page and return its DataFrame.
    """
    base_filename = os.path.splitext(os.path.basename(img_path))[0]
    
    original_pil = Image.open(img_path).convert('RGB')
    bgr_img = cv2.imread(img_path)
    if bgr_img is None:
        return None
    
    h_orig, w_orig = bgr_img.shape[:2]
    target_size = (3200//2, 2496//2)
    preprocessed = preprocess(bgr_img, target_size)
    h_resized, w_resized = preprocessed.shape[:2]

    img_tensor = torch.from_numpy(preprocessed).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).cuda()

    with torch.no_grad():
        logits = model_img(img_tensor)['out']
    final_mask = post_process_mask(logits, 1000, 40)
    
    if final_mask.dim() == 3:
        final_mask = final_mask[0]

    bboxes = extract_row_col_bboxes(final_mask)
    scaled_bboxes = [
        scale_bbox(bbox, (w_orig, h_orig), (w_resized, h_resized))
        for bbox in bboxes
    ]
    
    outputs = []
    for row_num, col_num, x1, y1, x2, y2 in scaled_bboxes:
        crop = original_pil.crop((x1, y1, x2, y2))
        
        try:
            pixel_values = process_image_for_ocr(crop, processor)
            pixel_values = pixel_values.to(model_ocr.device)

            generated_ids = model_ocr.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                do_sample=True,
                temperature=1.0,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.0,
                length_penalty=1.0,
                no_repeat_ngram_size=3
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            outputs.append((row_num, col_num, generated_text))
            
            # saving crop for debugging
            crop_path = os.path.join(out_dir, f"{base_filename}_row_{row_num}_col_{col_num}.png")
            crop.save(crop_path)
            
        except Exception as e:
            print(f"Error processing region (row {row_num}, col {col_num}): {str(e)}")
            continue
    
    outputs.sort()
    return create_ocr_matrix(outputs)

def scale_bbox(bbox, original_size, resized_size):
    """
    Scale bounding box coordinates from resized to original image dimensions
    """
    row_num, col_num, x1, y1, x2, y2 = bbox
    w_orig, h_orig = original_size
    w_resized, h_resized = resized_size
    
    x1_orig = int(x1 * (w_orig / w_resized))
    y1_orig = int(y1 * (h_orig / h_resized))
    x2_orig = int(x2 * (w_orig / w_resized))
    y2_orig = int(y2 * (h_orig / h_resized))
    
    return (row_num, col_num, x1_orig, y1_orig, x2_orig, y2_orig)

def process_image_for_ocr(image, processor, min_size=32):
    """
    Preprocess image for OCR with size checking and padding
    """
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    width, height = image.size
    
    # add padding if image is too small
    if width < min_size or height < min_size:
        new_width = max(width, min_size)
        new_height = max(height, min_size)
        padded_image = Image.new('RGB', (new_width, new_height), (255, 255, 255))
        padded_image.paste(image, (0, 0))
        image = padded_image
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values

def create_ocr_matrix(outputs):
    """
    Convert OCR outputs into a structured matrix/dataframe.
    """
    if not outputs:
        return pd.DataFrame()
    
    max_row = max(row for row, _, _ in outputs) + 1
    max_col = max(col for _, col, _ in outputs) + 1
    
    matrix = np.full((max_row, max_col), np.nan, dtype=object)
    for row_num, col_num, text in outputs:
        matrix[row_num, col_num] = text
    
    column_names = {
        0: "Year",
        1: "Date",
        2: "Latitude",
        3: "Longitude",
        4: "Temperature"
    }
    
    df = pd.DataFrame(matrix, columns=[column_names.get(i, f"Column_{i+1}") for i in range(max_col)])
    
    return df

def save_matrix_outputs(df, output_path, base_filename):
    """
    Save matrix as csv
    """
    os.makedirs(output_path, exist_ok=True)
    
    csv_path = os.path.join(output_path, f"{base_filename}_matrix.csv")
    df.to_csv(csv_path, index=False)
    
    return csv_path

def process_and_merge_pages(data_dir, out_dir, matrix_dir, model_img, model_ocr, processor, single_debug=False, multi_debug=False):
    """
    Process and merge pages with options for single or multi debug mode.
    
    Args:
        data_dir: Directory containing the page images
        out_dir: Directory for output crops
        matrix_dir: Directory for matrix outputs
        model_img: Image segmentation model
        model_ocr: OCR model
        processor: OCR processor
        single_debug: If True, process only one pair of pages
        multi_debug: If True, process only three pairs of pages
        
    Returns:
        list: List of paths to generated CSV files
    """
    page_pairs = find_matching_pages(data_dir)
    
    if single_debug:
        if page_pairs:
            page_pairs = [page_pairs[0]]  # take only the first pair
        else:
            print("No page pairs found!")
            return []
    elif multi_debug:
        if len(page_pairs) >= 3:
            page_pairs = page_pairs[:3]  # take only the first three pairs
        else:
            print(f"Warning: Only found {len(page_pairs)} pairs, using all available")
    
    generated_csvs = []
    
    for top_path, bottom_path in page_pairs:
        print(f"\nProcessing page pair:")
        print(f"Top: {os.path.basename(top_path)}")
        print(f"Bottom: {os.path.basename(bottom_path)}")
        
        top_df = process_single_page(top_path, model_img, model_ocr, processor, out_dir)
        if top_df is None:
            print(f"Error processing top page: {top_path}")
            continue
            
        bottom_df = process_single_page(bottom_path, model_img, model_ocr, processor, out_dir)
        if bottom_df is None:
            print(f"Error processing bottom page: {bottom_path}")
            continue
        
        merged_df = merge_page_dataframes(top_df, bottom_df)
        
        base_filename = os.path.splitext(os.path.basename(top_path))[0].replace('-t', '')
        csv_path = save_matrix_outputs(merged_df, matrix_dir, base_filename)
        generated_csvs.append(csv_path)
        
        print(f"\nOutput saved to:")
        print(f"CSV: {csv_path}")
    
    return generated_csvs

def merge_all_csvs(csv_paths, output_dir):
    """
    Merge multiple CSV files into a single database.
    
    Args:
        csv_paths: List of paths to CSV files to merge
        output_dir: Directory to save the merged database
        
    Returns:
        str: Path to the merged database file
    """
    if not csv_paths:
        print("No CSV files to merge!")
        return None
        
    dfs = []
    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
            df['source_file'] = os.path.basename(csv_path)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {csv_path}: {str(e)}")
    
    if not dfs:
        print("No valid DataFrames to merge!")
        return None
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Save merged database
    merged_path = os.path.join(output_dir, "0_merged_database.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"\nMerged database saved to: {merged_path}")
    print(f"Total entries: {len(merged_df)}")
    
    return merged_path

def main():
    parser = argparse.ArgumentParser(description='Process logbook pages with OCR')
    parser.add_argument('--single-debug', action='store_true',
                      help='Process only one pair of pages for debugging')
    parser.add_argument('--multi-debug', action='store_true',
                      help='Process only three pairs of pages and merge them')
    parser.add_argument('--merge', action='store_true',
                      help='Merge all processed pages into a single database')
    parser.add_argument('--data-dir', type=str, default="data/processed/Mathiesen-single-pages",
                      help='Directory containing the page images')
    parser.add_argument('--results-dir', type=str, default="results/ocr",
                      help='Root directory for all outputs')
    
    args = parser.parse_args()
    
    random.seed(42)
    
    out_dir = os.path.join(args.results_dir, "row_col_crops")
    matrix_dir = os.path.join(args.results_dir, "matrices")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(matrix_dir, exist_ok=True)

    model_img = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.DEFAULT)    
    model_img.classifier[-1] = torch.nn.Conv2d(256, 6, kernel_size=1)
    model_img.load_state_dict(torch.load('results/DeepLab3_ResNet50_Tversky_250/best_model.pth', weights_only=True))
    model_img.cuda()
    model_img.eval()

    model_ocr = VisionEncoderDecoderModel.from_pretrained("results/trocr_finetuned_1250/final_model")
    processor = TrOCRProcessor.from_pretrained("results/trocr_finetuned_1250/final_model")
    model_ocr.to('cuda')

    if args.single_debug:
        generated_csvs = process_and_merge_pages(
            args.data_dir,
            out_dir,
            matrix_dir,
            model_img,
            model_ocr,
            processor,
            single_debug=True
        )
    elif args.multi_debug:
        generated_csvs = process_and_merge_pages(
            args.data_dir,
            out_dir,
            matrix_dir,
            model_img,
            model_ocr,
            processor,
            multi_debug=True
        )
        if generated_csvs:
            merge_all_csvs(generated_csvs, matrix_dir)
    else:
        generated_csvs = process_and_merge_pages(
            args.data_dir,
            out_dir,
            matrix_dir,
            model_img,
            model_ocr,
            processor
        )
        if args.merge and generated_csvs:
            merge_all_csvs(generated_csvs, matrix_dir)

    print("Processing complete!")


if __name__ == "__main__":
    main()