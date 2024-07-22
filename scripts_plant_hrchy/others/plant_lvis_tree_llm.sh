#!/bin/bash

conda activate shine

cd shine_cls || exit

echo "Planting LVIS LLM hierarchy tree to: shine/lvis_llm_answers"

CUDA_VISIBLE_DEVICES=0 python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode query \
                  --dataset_name lvis \
                  --output_root lvis_llm_answers

CUDA_VISIBLE_DEVICES=0 python -W ignore plant_llm_syn_hrchy_tree.py \
                  --mode postprocess \
                  --dataset_name lvis \
                  --output_root lvis_llm_answers