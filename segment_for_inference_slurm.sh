#!/bin/bash
#SBATCH --job-name=mgr-segment-for-inference-unet
#SBATCH --cpus-per-task=10 --mem=64G --exclude=hendrixgpu01fl,hendrixgpu05fl,hendrixgpu06fl,hendrixgpu09fl,hendrixgpu10fl,hendrixgpu17fl,hendrixgpu18fl,hendrixgpu19fl,hendrixgpu20fl
#SBATCH -p gpu --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%j.out

#. /etc/profile.d/modules.sh
#module load anaconda3/5.3.1

img_csv=${1:-images.csv}
model_dir=${2:-./}

python3 inference.py \
  --output_dir="out" \
  --model_dir=$model_dir \
  --img_csv=$img_csv
  --save_predict \
  --save_cutouts
