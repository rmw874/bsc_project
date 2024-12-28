#!/bin/bash
#SBATCH --job-name=big_and_greedy_a3b7
#SBATCH --output=slurm_logs/%j_%x.out        
#SBATCH --cpus-per-task=8                    
#SBATCH --mem=32G                            

# Specify GPU allocation on hendrixgpu16fl only - H100
#SBATCH --gres=gpu:1
#SBATCH --nodelist=hendrixgpu16fl

# Load Conda
source /home/rmw874/miniforge3/etc/profile.d/conda.sh 

# Activate the 'piratbog' environment
conda activate piratbog

echo "Running on node: $(hostname)"
nvidia-smi

# Run your Python script
echo "Starting training script"
python -u /home/rmw874/piratbog/scripts/train_big_model.py
