#!/bin/bash
#SBATCH --job-name=overfitting-shiplog-unet
#SBATCH --cpus-per-task=10 --mem=64G --exclude=hendrixgpu01fl,hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl
#SBATCH -p gpu --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out

#. /etc/profile.d/modules.sh
#module load anaconda3/5.3.1

python overfitting.py