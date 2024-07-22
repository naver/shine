#!/bin/bash

conda activate shine

cd shine_cls || exit

echo "Planting ImageNet-1k LLM-generated hierarchy ground-truth hierarchy tree to: shine_cls/imagenet1k"

CUDA_VISIBLE_DEVICES=0 python -W ignore plant_hierarchy.py --source "chatgpt"

CUDA_VISIBLE_DEVICES=0 python -W ignore plant_hierarchy.py --source "chatgpt_post"