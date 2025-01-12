import os

# Training parameters
BATCH_SIZE = 4
EPOCHS = 250
LEARNING_RATE = 1e-4
N_CLASSES = 6
TARGET_SIZE = (3200 // 2, 2496 // 2)
RUN_NAME = "segmentation_model"

# Data paths
DATA_ROOT = "data/processed"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train/images")
TRAIN_MASK_DIR = os.path.join(DATA_ROOT, "train/masks")
VAL_IMG_DIR = os.path.join(DATA_ROOT, "val/images")
VAL_MASK_DIR = os.path.join(DATA_ROOT, "val/masks")

# Model save/load paths
RESULTS_DIR = f"results/{RUN_NAME}"
BEST_MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.pth")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Loss function parameters
TVERSKY_PARAMS = {
    'alpha': 0.3,
    'beta': 0.7,
}

# Post-processing parameters
POST_PROCESSING = {
    'min_region_size': 1000,
    'expected_row_height': 40,
}

# DataLoader parameters
DATALOADER_PARAMS = {
    'batch_size': BATCH_SIZE,
    'num_workers': 6,
    'pin_memory': True,
    'persistent_workers': True,
    'prefetch_factor': 2
}

# Visualization parameters
VISUALIZATION = {
    'class_labels': ["Year", "Date", "Latitude", "Longitude", "Temperature", "Background"],
    'figure_size': (20, 15)
}

# Visualization frequency
TRAIN_VIZ_FREQ = 50  # Visualize every N epochs during training
VAL_VIZ_FREQ = 10   # Visualize every N epochs during validation

# Device configuration
CUDA_LAUNCH_BLOCKING = "1"
PYTORCH_CUDA_ALLOC_CONF = "garbage_collection_threshold:0.75"