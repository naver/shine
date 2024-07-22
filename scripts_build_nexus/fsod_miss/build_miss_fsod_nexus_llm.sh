#!/bin/bash

activateEnvironmentAndMove() {
    # Activate the conda environment
    conda activate shine

    # Change directory
    cd shine || exit
}

buildNexus() {
    declare -A nexus_paths=(
        ["ViT-B/32"]="../nexus/fsod_miss/vitB32/shine_llm"
    )

    for clip_model in "${!nexus_paths[@]}"; do
        CUDA_VISIBLE_DEVICES=0 python -W ignore build_miss_inat_fsod_aggr_w_llm_hrchy.py \
                          --dataset_name "fsod_expanded" \
                          --gpt_results_root "fsod_llm_answers" \
                          --prompter isa \
                          --aggregator mean \
                          --clip_model "$clip_model" \
                          --out_path "${nexus_paths[$clip_model]}"
    done
}

# Main script execution
activateEnvironmentAndMove
buildNexus
