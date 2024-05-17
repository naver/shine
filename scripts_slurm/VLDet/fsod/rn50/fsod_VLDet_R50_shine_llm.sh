#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/fsod_VLDet_R50_shine_llm.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

# Configuration files
CFG_VL_R50="configs_vldet/VLDet_LbaseCCcap_CLIP_R5021k_640b64_2x_ft4x_caption.yaml"
# Model weight files
W_VL_R50="models/vldet/lvis_vldet.pth"

METADATA_ROOT="nexus/fsod/rn50/shine_llm"

python train_net_vldet.py \
        --num-gpus 8 \
        --config-file ${CFG_VL_R50} \
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ${W_VL_R50} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False
