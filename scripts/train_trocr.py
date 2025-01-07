import os
import torch
import json
import shutil
from pathlib import Path
import random
import numpy as np
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import Dataset

class OCRDataset(Dataset):
    def __init__(self, data_dir, processor):
        self.data_dir = Path(data_dir)
        self.processor = processor
        
        # Load annotations
        with open(self.data_dir / "annotations.json") as f:
            self.annotations = json.load(f)
            
        self.examples = list(self.annotations.items())
        
        # Cache for processed images
        self.cache = {}
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        filename, text = self.examples[idx]
        
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
        
        # Load and process image
        image_path = self.data_dir / filename
        image = Image.open(image_path).convert("RGB")
        
        # Process image
        pixel_values = np.array(
            self.processor(
                image, 
                return_tensors=None
            ).pixel_values
        )
        
        # Process text
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)
        
        # Convert to torch tensor immediately
        pixel_values = torch.from_numpy(pixel_values)
        
        # Cache the processed data
        item = {
            "pixel_values": pixel_values,
            "labels": labels
        }
        self.cache[idx] = item
        
        return item

def collate_fn(examples):
    # Stack pre-converted tensors
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels,
    }

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        # Remove num_items_in_batch from inputs if it exists
        if isinstance(inputs, dict) and "num_items_in_batch" in inputs:
            inputs = {k: v for k, v in inputs.items() if k != "num_items_in_batch"}
            
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
        
    def log(self, logs):
        """Override logging to show progress bar"""
        logs = super().log(logs)
        
        if hasattr(self, 'epoch_pbar'):
            self.epoch_pbar.update(1)
        if hasattr(self, 'step_pbar'):
            self.step_pbar.update(1)
            
            if 'loss' in logs:
                self.step_pbar.set_postfix({
                    'loss': f"{logs['loss']:.4f}",
                    'lr': f"{logs.get('learning_rate', 0):.2e}"
                })
        
        return logs
    
    def _setup_progress_bars(self):
        """Setup progress bars for training"""
        self.epoch_pbar = tqdm(total=self.args.num_train_epochs, desc="Epochs")
        total_steps = len(self.train_dataset) // (self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps)
        self.step_pbar = tqdm(total=total_steps, desc="Training steps")

def train_trocr():
    # Prepare data
    print("Preparing datasets...")
    
    # Set deterministic operations for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_dir = Path("data/ocr_training_crops")
    val_dir = Path("data/ocr_training_crops")  # Using same dir for simplicity
    
    # Initialize model and processor
    print("Loading model and processor...")
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = OCRDataset(train_dir, processor)
    val_dataset = OCRDataset(val_dir, processor)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="results/trocr_finetuned",
        num_train_epochs=15,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        fp16=True,
        gradient_accumulation_steps=2,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        learning_rate=5e-5,
        warmup_ratio=0.1,  # Warmup for 10% of steps
        weight_decay=0.01,
        gradient_checkpointing=True,
        report_to=[],  # Disable wandb
        # Memory optimizations
        dataloader_num_workers=2,  # Reduced workers
        dataloader_pin_memory=True,
        # Other settings
        disable_tqdm=True,
        seed=42,
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    # Setup progress bars
    trainer._setup_progress_bars()
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Close progress bars
    if hasattr(trainer, 'epoch_pbar'):
        trainer.epoch_pbar.close()
    if hasattr(trainer, 'step_pbar'):
        trainer.step_pbar.close()
    
    # Save final model
    print("Saving model...")
    model.save_pretrained("results/trocr_finetuned/final_model")
    processor.save_pretrained("results/trocr_finetuned/final_model")
    
    print("Training complete!")

if __name__ == "__main__":
    train_trocr()