import torch
import json
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd

def compute_cer(prediction, ground_truth):
    """
    Compute Character Error Rate using Levenshtein distance.
    """
    if len(ground_truth) == 0:
        return 1.0 if len(prediction) > 0 else 0.0
        
    dp = [[0] * (len(ground_truth) + 1) for _ in range(len(prediction) + 1)]
    
    for i in range(len(prediction) + 1):
        dp[i][0] = i
    for j in range(len(ground_truth) + 1):
        dp[0][j] = j
        
    for i in range(1, len(prediction) + 1):
        for j in range(1, len(ground_truth) + 1):
            if prediction[i-1] == ground_truth[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # deletion
                    dp[i][j-1] + 1,    # insertion
                    dp[i-1][j-1] + 1   # substitution
                )
    
    return dp[len(prediction)][len(ground_truth)] / len(ground_truth)

def plot_cer_distribution(predictions, output_dir):
    """
    Create visualizations of CER distribution and error analysis.
    
    Args:
        predictions: List of dictionaries containing prediction data
        output_dir: Directory to save plots
    """
    # Convert predictions to DataFrame for easier analysis
    df = pd.DataFrame(predictions)
    
    # 1. CER Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='cer', bins=30, kde=True)
    plt.title('Distribution of Character Error Rates')
    plt.xlabel('Character Error Rate')
    plt.ylabel('Count')
    plt.axvline(df['cer'].mean(), color='r', linestyle='--', 
                label=f'Mean CER: {df["cer"].mean():.3f}')
    plt.legend()
    plt.savefig(output_dir / 'cer_distribution.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 2. Error vs Text Length Scatter Plot
    df['gt_length'] = df['ground_truth'].str.len()
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='gt_length', y='cer', alpha=0.5)
    plt.title('Character Error Rate vs Text Length')
    plt.xlabel('Ground Truth Length')
    plt.ylabel('Character Error Rate')
    
    # Add trend line
    z = np.polyfit(df['gt_length'], df['cer'], 1)
    p = np.poly1d(z)
    plt.plot(df['gt_length'], p(df['gt_length']), "r--", 
             label=f'Trend line (slope: {z[0]:.4f})')
    plt.legend()
    plt.savefig(output_dir / 'cer_vs_length.png', bbox_inches='tight', dpi=150)
    plt.close()

    # 3. Worst Cases Analysis
    n_worst = 10
    worst_cases = df.nlargest(n_worst, 'cer')
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=worst_cases, x=worst_cases.index, y='cer')
    plt.title(f'Top {n_worst} Worst Character Error Rates')
    plt.xticks(rotation=45)
    plt.xlabel('Example Index')
    plt.ylabel('Character Error Rate')
    plt.savefig(output_dir / 'worst_cases.png', bbox_inches='tight', dpi=150)
    plt.close()

    # Save worst cases details
    worst_cases_file = output_dir / 'worst_cases.txt'
    with open(worst_cases_file, 'w') as f:
        f.write("Analysis of Worst Performing Cases:\n\n")
        for _, row in worst_cases.iterrows():
            f.write(f"File: {row['filename']}\n")
            f.write(f"Ground Truth: '{row['ground_truth']}'\n")
            f.write(f"Prediction: '{row['prediction']}'\n")
            f.write(f"CER: {row['cer']:.4f}\n")
            f.write("-" * 50 + "\n")

    # 4. Error Rate Summary Statistics
    summary_stats = {
        'mean_cer': df['cer'].mean(),
        'median_cer': df['cer'].median(),
        'std_cer': df['cer'].std(),
        'min_cer': df['cer'].min(),
        'max_cer': df['cer'].max(),
        'percentile_25': df['cer'].quantile(0.25),
        'percentile_75': df['cer'].quantile(0.75)
    }
    
    # Create summary statistics plot
    plt.figure(figsize=(10, 6))
    plt.bar(summary_stats.keys(), summary_stats.values())
    plt.title('CER Summary Statistics')
    plt.xticks(rotation=45)
    plt.ylabel('Value')
    plt.savefig(output_dir / 'summary_statistics.png', bbox_inches='tight', dpi=150)
    plt.close()

    return summary_stats

class ValidationDataset(Dataset):
    def __init__(self, data_dir, processor, max_length=128):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        
        with open(self.data_dir / "annotations.json") as f:
            self.annotations = json.load(f)
        self.examples = list(self.annotations.items())
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        filename, text = self.examples[idx]
        image_path = self.data_dir / filename
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        
        return {
            "pixel_values": pixel_values.squeeze(),
            "text": text,
            "filename": filename
        }

def validate_trocr(model_path, validation_dir, batch_size=8):
    """
    Validate TrOCR model and compute CER on validation set with visualizations.
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = TrOCRProcessor.from_pretrained(model_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()
    
    val_dataset = ValidationDataset(validation_dir, processor)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    predictions = []
    
    # Validation loop
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            pixel_values = batch["pixel_values"].to(device)
            texts = batch["text"]
            filenames = batch["filename"]
            
            generated_ids = model.generate(
                pixel_values,
                max_length=128,
                num_beams=4,
                temperature=1.0,
                top_k=50,
                top_p=0.95
            )
            
            pred_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for pred, truth, fname in zip(pred_texts, texts, filenames):
                cer = compute_cer(pred, truth)
                predictions.append({
                    "filename": fname,
                    "prediction": pred,
                    "ground_truth": truth,
                    "cer": cer
                })
    
    # Create output directory
    output_dir = Path("results/trocr_validation")
    output_dir.mkdir(exist_ok=True)
    
    # Generate visualizations and get summary statistics
    summary_stats = plot_cer_distribution(predictions, output_dir)
    
    # Save all results
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump({
            "summary_statistics": summary_stats,
            "predictions": predictions
        }, f, indent=2)
    
    return summary_stats, predictions

if __name__ == "__main__":
    model_path = "results/trocr_finetuned_1250/final_model"
    validation_dir = "data/ocr_validation_crops"
    
    summary_stats, predictions = validate_trocr(
        model_path,
        validation_dir,
        batch_size=8
    )
    
    print("\nValidation Summary:")
    print(f"Mean CER: {summary_stats['mean_cer']:.4f}")
    print(f"Median CER: {summary_stats['median_cer']:.4f}")
    print(f"Std Dev: {summary_stats['std_cer']:.4f}")
    print(f"25th Percentile: {summary_stats['percentile_25']:.4f}")
    print(f"75th Percentile: {summary_stats['percentile_75']:.4f}")