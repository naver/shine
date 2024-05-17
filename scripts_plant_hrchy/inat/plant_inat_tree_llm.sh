#!/bin/bash

# Activate the conda environment
conda activate shine

# Change directory
cd shine_cls || exit

# Announcement
echo "Planting iNat LLM hierarchy tree to: shine/inat_llm_answers"

# Define the hierarchy levels
h_levels=(l1 l2 l3 l4 l5 l6)

# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Querying for iNat ${level} super-/sub-categories..."

    python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode query \
           --dataset_name inat \
           --output_root inat_llm_answers \
           --h_level "$level"

    echo "Saved the quried results to: shine/inat_llm_answers/raw_inat_gpt_hrchy_${level}.json"
done


# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Cleaning for iNat ${level} LLM query results..."

    python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode postprocess \
           --dataset_name inat \
           --output_root inat_llm_answers \
           --h_level "$level"

    echo "Saved the cleaned results to: shine/inat_llm_answers/cleaned_inat_gpt_hrchy_${level}.json"
done