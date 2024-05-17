#!/bin/bash

conda activate shine

cd shine_cls || exit

echo "Planting FSOD ground-truth hierarchy tree to: shine/fsod_annotations/fsod_hierarchy_tree.json"

python -W ignore plant_fsod_hrchy_tree.py