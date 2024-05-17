#!/bin/bash

conda activate shine

# Configuration files
CFG_VL_R50="configs_vldet/VLDet_LbaseCCcap_CLIP_R5021k_640b64_2x_ft4x_caption.yaml"
# Model weight files
W_VL_R50="models/vldet/lvis_vldet.pth"

METADATA_ROOT="nexus/inat/rn50/shine_llm"

python train_net_vldet.py \
        --num-gpus 1 \
        --config-file ${CFG_VL_R50} \
        --eval-only \
        DATASETS.TEST "('inat_val_l1', 'inat_val_l2', 'inat_val_l3','inat_val_l4','inat_val_l5','inat_val_l6',)" \
        MODEL.WEIGHTS ${W_VL_R50} \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/inat_clip_hrchy_l1.npy', '${METADATA_ROOT}/inat_clip_hrchy_l2.npy', '${METADATA_ROOT}/inat_clip_hrchy_l3.npy', '${METADATA_ROOT}/inat_clip_hrchy_l4.npy', '${METADATA_ROOT}/inat_clip_hrchy_l5.npy', '${METADATA_ROOT}/inat_clip_hrchy_l6.npy',)" \
        MODEL.TEST_NUM_CLASSES "(5, 18, 61, 184, 317, 500,)" \
        MODEL.MASK_ON False
