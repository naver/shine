#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/fsod_miss_detic_SwinB_LVIS_baseline.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

METADATA_ROOT="nexus/fsod_miss/vitB32/baseline"

python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/BoxSup-C2_L_CLIP_SwinB_896b32_4x.yaml\
        --eval-only \
        DATASETS.TEST "('fsod_test_l3',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/BoxSup-C2_L_CLIP_SwinB_896b32_4x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "(${METADATA_ROOT}/fsod_clip_hrchy_l3_over_oid_lvis.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1570,)" \
        MODEL.MASK_ON False
