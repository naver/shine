#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:1
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/cls_imagenet1k_vitL14_shine_wordnet.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

activateAndRun() {
    # Activate the conda environment
    conda activate shine

    # Change to the specified directory, exit if it fails
    cd shine_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-L/14" \
              --method "shine" \
              --hierarchy_tree_path "imagenet1k_hrchy_wordnet.json" \
              --batch_size 256 \
              --num_runs 1
}

# Call the function
activateAndRun

