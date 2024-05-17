#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/coco_ovod_BoxSup_CLIP_R50_1x_shine_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

METADATA_ROOT="nexus/coco/shine_llm"


python train_net_detic_coco.py \
        --num-gpus 8 \
        --config-file ./configs_detic/coco/BoxSup_OVCOCO_CLIP_R50_1x.yaml \
        --eval-only \
        MODEL.WEIGHTS ./models/detic/coco_ovod/BoxSup_OVCOCO_CLIP_R50_1x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/coco_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False