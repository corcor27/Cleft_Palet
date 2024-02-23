#!/bin/bash --login
#$ -cwd
#SBATCH --job-name=DDSM_Breast_Cancer
#SBATCH --out=cleft_model.out.%J
#SBATCH --err=cleft_model.err.%J
#SBATCH -p gpu
#SBATCH --gres=gpu:1

python Custom_3D_to_2D.py
