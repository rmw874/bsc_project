#!/bin/bash
#SBATCH --job-name=unet_training       # Job name
#SBATCH --output=logs/%x_%j.out        # Output log file (%x = job name, %j = job ID)
#SBATCH --error=logs/%x_%j.err         # Error log file
#SBATCH --partition=gpu                # Partition to submit to (use "gpu" for GPU nodes)
#SBATCH --gres=gpu:1                   # Number of GPUs (adjust as needed)
#SBATCH --cpus-per-task=10             # Number of CPU cores per task
#SBATCH --mem=64G                      # Memory per node (adjust as needed)
#SBATCH --mail-type=END,FAIL           # Send email on completion or failure
#SBATCH --mail-user=heiberg.oscar@gmail.com  # Email for notifications

#. /etc/profile.d/modules.sh
#module load anaconda3/5.3.1

module load python/3.11                # Load necessary modules
module load tensorflow/2.11.0          # Load TensorFlow module
source /path/to/your/venv/bin/activate # Activate virtual environment

# Run your training script
python scripts/train_model.py --config config/config.yaml
