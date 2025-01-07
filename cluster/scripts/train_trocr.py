import os
from pathlib import Path
import json
import torch
from PIL import Image
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from torch.utils.data import Dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class OCRDataset(Dataset):
    def __init__(self, image_dir: Path, processor, max_length: int = 64):
        self.image_dir = image_dir
        self.processor = processor
        self.max_length = max_length
        self.examples = []
        self.load_annotations()
    
    def load_annotations(self):
        json_path = self.image_dir / "annotations.json"
        with open(json_path, 'r') as f:
            self.annotations = json.load(f)
        
        self.examples = [
            {
                "image_path": str(self.image_dir / img_name),
                "text": text
            }
            for img_name, text in self.annotations.items()
            if os.path.exists(self.image_dir / img_name)
        ]
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Load and process image
        image = Image.open(example["image_path"]).convert("RGB")
        
        # Process image and ensure correct shape
        processed = self.processor(image, return_tensors="pt")
        pixel_values = processed.pixel_values.squeeze()  # Remove batch dimension
        
        # Process text
        encoded = self.processor.tokenizer(
            example["text"],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        labels = encoded.input_ids.squeeze()  # Remove batch dimension
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def collate_fn(examples):
    # Collate images
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.stack([example["labels"] for example in examples])
    
    return {
        "pixel_values": pixel_values,
        "labels": labels
    }

class CustomTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        """
        # Remove num_items_in_batch from inputs if it exists
        if isinstance(inputs, dict) and "num_items_in_batch" in inputs:
            del inputs["num_items_in_batch"]
            
        outputs = model(**inputs)
        loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss

def main():
    # Set paths
    train_dir = Path("data/ocr_training_crops")
    val_dir = Path("data/ocr_validation_crops")
    
    # Load model and processor
    model_name = "microsoft/trocr-base-handwritten"
    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Configure decoder params
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # Prepare datasets
    train_dataset = OCRDataset(train_dir, processor)
    val_dataset = OCRDataset(val_dir, processor)
    
    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")
    
    # Let's verify the data shape
    sample_batch = collate_fn([train_dataset[0], train_dataset[1]])
    print(f"\nSample batch shapes:")
    print(f"Pixel values: {sample_batch['pixel_values'].shape}")
    print(f"Labels: {sample_batch['labels'].shape}")
    
    # Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir="results/trocr_finetuned_1250",
        num_train_epochs=75,                     
        per_device_train_batch_size=32,         
        per_device_eval_batch_size=32,           
        fp16=True,
        gradient_accumulation_steps=1,           
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,                      
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        gradient_checkpointing=False,            
        dataloader_num_workers=4,                
        dataloader_pin_memory=True,
        generation_max_length=64,
        predict_with_generate=True,
        remove_unused_columns=False  
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    print("\nSaving model...")
    trainer.save_model("results/trocr_finetuned_1250/final_model")
    processor.save_pretrained("results/trocr_finetuned_1250/final_model")
    
    print("Training complete!")

if __name__ == "__main__":
    main()