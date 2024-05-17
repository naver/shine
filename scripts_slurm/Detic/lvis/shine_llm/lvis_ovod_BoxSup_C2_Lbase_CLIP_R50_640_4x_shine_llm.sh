#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/lvis_ovod_BoxSup_C2_Lbase_CLIP_R50_640_4x_shine_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

METADATA_ROOT="nexus/lvis/shine_llm"

python train_net_detic.py \
        --num-gpus 8 \
        --config-file ./configs_detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.yaml\
        --eval-only \
        DATASETS.TEST "('lvis_v1_val',)" \
        MODEL.WEIGHTS ./models/detic/lvis_ovod/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/lvis_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1203,)" \
        MODEL.MASK_ON False
