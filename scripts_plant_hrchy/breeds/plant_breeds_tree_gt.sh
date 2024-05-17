#!/bin/bash

conda activate shine

cd shine_cls || exit

echo "Planting BREEDS hierarchy ground-truth hierarchy tree to: shine_cls/hrchy_breeds"

python -W ignore plant_hierarchy.py --source "breeds"