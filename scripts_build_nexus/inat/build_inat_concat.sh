#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate shine

    # Change directory
    cd shine || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]="../nexus/inat/vitB32/concat"
        ["RN50"]="../nexus/inat/rn50/concat"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -W ignore build_inat_sing.py \
                          --prompter concat \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus
