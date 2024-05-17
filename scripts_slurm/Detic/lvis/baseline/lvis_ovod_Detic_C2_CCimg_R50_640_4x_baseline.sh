#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/lvis_ovod_Detic_C2_CCimg_R50_640_4x_baseline.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

METADATA_ROOT="nexus/lvis/baseline"

python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('lvis_v1_val',)" \
        MODEL.WEIGHTS ./models/detic/lvis_ovod/Detic_LbaseCCimg_CLIP_R5021k_640b64_4x_ft4x_max-size.pth\
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/lvis_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1203,)" \
        MODEL.MASK_ON False
