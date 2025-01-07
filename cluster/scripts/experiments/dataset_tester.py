from torch.utils.data import DataLoader
from dataset import PirateLogDataset

# Parameters
BATCH_SIZE = 8
TARGET_SIZE = (3200//2, 2496//2)
dataset = PirateLogDataset(img_dir='data/processed/images', mask_dir='data/processed/masks', target_size=TARGET_SIZE, num_classes=6)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)

# Test loading batches
for i, (img_tensor, mask_tensor) in enumerate(dataloader):
    print(f"Batch {i}: Image tensor size {img_tensor.size()}, Mask tensor size {mask_tensor.size()}")
    if i > 6:  # Exit after a few batches
        break