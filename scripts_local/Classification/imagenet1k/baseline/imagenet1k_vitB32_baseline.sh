#!/bin/bash

activateAndRun() {
    # Activate the conda environment
    conda activate shine

    # Change to the specified directory, exit if it fails
    cd shine_cls || exit

   # If you wanna test inference speed, change --num_runs to 10
    python -W ignore zeroshot.py \
              --model_size "ViT-B/32" \
              --method "zeroshot" \
              --batch_size 64 \
              --num_runs 1
}

# Call the function
activateAndRun
