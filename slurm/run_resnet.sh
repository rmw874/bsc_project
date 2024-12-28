#!/bin/bash
#SBATCH --job-name=resnet_tversky
#SBATCH --output=slurm_logs/%j_%x.out        
#SBATCH --cpus-per-task=8                    
#SBATCH --mem=16G                            
#SBATCH --gres=gpu:1                 
#SBATCH --exclude=hendrixgpu10fl,hendrixgpu09fl,hendrixgpu19fl,hendrixgpu20fl,hendrixgpu17fl

# Load Conda
source /home/rmw874/miniforge3/etc/profile.d/conda.sh 

# Activate the 'piratbog' environment
conda activate piratbog

echo "Running on node: $(hostname)"
nvidia-smi

# Run your Python script
echo "Starting training script"
python -u /home/rmw874/piratbog/scripts/resnet_train.py