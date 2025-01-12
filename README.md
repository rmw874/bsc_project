# Ship Logbook Digitization Pipeline

This project implements an end-to-end pipeline for digitizing historical ship logbooks, achieving 86.7% accuracy in extracting and transcribing tabular entries.

## Features

- Semantic segmentation using DeepLabV3 with ResNet50 backbone
- Custom post-processing pipeline for cell separation
- TrOCR-based handwritten text recognition
- Preprocessing optimized for degraded historical documents

## Requirements

- Python 3.8+
- CUDA-capable GPU with 8GB+ memory (ideally 16GB+)
- See `requirements.txt` for Python package dependencies

## Setup

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download pretrained models:

```bash
mkdir -p results/
# Install gdown if not already installed
pip install gdown

# Download segmentation model
gdown --id 1q8MpVCo0FCOzLivz7Lddnuga88nbO85f -O results/DeepLab3_ResNet50_Tversky_350/best_model.pth

# Download OCR model
gdown --id 1z2e7VkfAraSoaW4fJaqfHyqLEnHKoLU8 -O results/trocr_finetuned_1250/final_model/
```

## Usage

### Training Segmentation Model

```python
python train.py
```

### OCR Fine-tuning

```python
python train_trocr.py
```

### Full Pipeline

```python
python imgs2csv.py --merge --data-dir path/to/images --results-dir path/to/output
```

## Project Structure

- `config.py` - Configuration parameters
- `dataset.py` - Data loading and preprocessing for segmentation
- `post_process.py` - Cell separation and cleaning
- `train.py` - Segmentation model training
- `train_trocr.py` - OCR model fine-tuning
- `imgs2csv.py` - End-to-end pipeline
