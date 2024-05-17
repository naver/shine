#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres gpu:8
#SBATCH --mem=64000
#SBATCH --time 15-00:00:00
#SBATCH --output=./slurm-output/fsod_codet_rn50_shine_gt.out

export PATH="/home/mliu/software/anaconda3/bin:$PATH"

eval "$(conda shell.bash hook)"
bash

conda activate shine

CFG_R50="./configs_codet/CoDet_OVLVIS_R5021k_4x_ft4x.yaml"
MODEL_R50="./models/codet/CoDet_OVLVIS_R5021k_4x_ft4x.pth"

METADATA_ROOT="nexus/fsod/vitB32/shine_gt"

python train_net_codet.py \
        --num-gpus 8 \
        --config-file ${CFG_R50} \
        --eval-only \
        DATASETS.TEST "('fsod_test_l1', 'fsod_test_l2', 'fsod_test_l3',)" \
        MODEL.WEIGHTS ${MODEL_R50} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/fsod_clip_hrchy_l1.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l2.npy', '${METADATA_ROOT}/fsod_clip_hrchy_l3.npy',)" \
        MODEL.TEST_NUM_CLASSES "(15, 46, 200,)" \
        MODEL.MASK_ON False

