#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate shine

    # Change directory
    cd shine || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]="../nexus/coco/shine_llm"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        python -W ignore build_miss_inat_fsod_aggr_w_llm_hrchy.py \
                          --dataset_name "coco" \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus
