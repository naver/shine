#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/coco_ovod_Detic_CLIP_image_R50_1x_baseline.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

METADATA_ROOT="nexus/coco/baseline"

python train_net_detic_coco.py \
        --num-gpus 8 \
        --config-file ./configs_detic/Detic_OVCOCO_CLIP_R50_1x_max-size.yaml\
        --eval-only \
        MODEL.WEIGHTS ./models/detic/coco_ovod/Detic_OVCOCO_CLIP_R50_1x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/coco_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False

