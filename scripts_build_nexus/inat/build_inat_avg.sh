#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate shine

    # Change directory
    cd shine || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]="../nexus/inat/vitB32/avg"
        ["RN50"]="../nexus/inat/rn50/avg"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_inat_sing.py \
                          --prompter avg \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus
