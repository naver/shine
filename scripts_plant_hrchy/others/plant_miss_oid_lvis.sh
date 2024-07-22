#!/bin/bash

# Activate the conda environment
conda activate shine

# Change directory
cd shine_cls || exit

# Announcement
echo "Planting OID+LVIS LLM hierarchy tree to: shine/miss_lvis_oid_llm_answers"

# Define the hierarchy levels
h_levels=(l1 l2 l3 l4 l5 l6)

# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Querying for OID+LVIS ${level} super-/sub-categories..."

    CUDA_VISIBLE_DEVICES=0 python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode query \
           --dataset_name oid_lvis \
           --output_root miss_lvis_oid_llm_answers \
           --h_level "$level"

    echo "Saved the quried results to: shine/miss_lvis_oid_llm_answers/raw_oid_lvis_gpt_hrchy_${level}.json"
done


# Loop through each hierarchy level
for level in "${h_levels[@]}"; do
    echo "Cleaning for OID+LVIS ${level} LLM query results..."

    CUDA_VISIBLE_DEVICES=0 python -W ignore plant_llm_syn_hrchy_tree.py \
           --mode postprocess \
           --dataset_name oid_lvis \
           --output_root miss_lvis_oid_llm_answers \
           --h_level "$level"

    echo "Saved the cleaned results to: shine/miss_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_hrchy_${level}.json"
done