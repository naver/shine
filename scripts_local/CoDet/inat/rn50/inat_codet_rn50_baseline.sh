#!/bin/bash

conda activate shine

CFG_R50="./configs_codet/CoDet_OVLVIS_R5021k_4x_ft4x.yaml"
MODEL_R50="./models/codet/CoDet_OVLVIS_R5021k_4x_ft4x.pth"

METADATA_ROOT="nexus/inat/vitB32/baseline"

python train_net_codet.py \
        --num-gpus 1 \
        --config-file ${CFG_R50} \
        --eval-only \
        DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)" \
        MODEL.WEIGHTS ${MODEL_R50} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l1.npy', '${METADATA_ROOT}/inat_clip_hrchy_l2.npy', '${METADATA_ROOT}/inat_clip_hrchy_l3.npy', '${METADATA_ROOT}/inat_clip_hrchy_l4.npy', '${METADATA_ROOT}/inat_clip_hrchy_l5.npy', '${METADATA_ROOT}/inat_clip_hrchy_l6.npy', )" \
        MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)" \
        MODEL.MASK_ON False

