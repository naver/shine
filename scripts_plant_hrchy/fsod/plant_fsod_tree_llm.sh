#!/bin/bash

# Activate the conda environment
conda activate shine

# Change directory
cd shine_cls || exit

# Announcement
echo "Planting FSOD LLM hierarchy tree to: shine/fsod_llm_answers"

# Define the hierarchy levels
h_levels=(l1 l2 l3)

# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Querying for FSOD ${level} super-/sub-categories..."

    python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode query \
           --dataset_name fsod \
           --output_root fsod_llm_answers \
           --h_level "$level"

    echo "Saved the quried results to: shine/fsod_llm_answers/raw_fsod_gpt_hrchy_${level}.json"
done


# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Cleaning for FSOD ${level} LLM query results..."

    python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode postprocess \
           --dataset_name fsod \
           --output_root fsod_llm_answers \
           --h_level "$level"

    echo "Saved the cleaned results to: shine/fsod_llm_answers/cleaned_fsod_gpt_hrchy_${level}.json"
done