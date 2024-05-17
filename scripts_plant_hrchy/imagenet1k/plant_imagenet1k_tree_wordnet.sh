#!/bin/bash

conda activate shine

cd shine_cls || exit

echo "Planting ImageNet-1k WordNet hierarchy ground-truth hierarchy tree to: shine_cls/imagenet1k"

python -W ignore plant_hierarchy.py --source "wordnet"