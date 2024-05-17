#!/bin/bash

conda activate shine

METADATA_ROOT="nexus/inat/vitB32/shine_llm"

python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/only_test_Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml \
        --eval-only \
        DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)" \
        MODEL.WEIGHTS ./models/detic/cross_eval/Detic_LI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l1.npy', '${METADATA_ROOT}/inat_clip_hrchy_l2.npy', '${METADATA_ROOT}/inat_clip_hrchy_l3.npy', '${METADATA_ROOT}/inat_clip_hrchy_l4.npy', '${METADATA_ROOT}/inat_clip_hrchy_l5.npy', '${METADATA_ROOT}/inat_clip_hrchy_l6.npy', )" \
        MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)" \
        MODEL.MASK_ON False
