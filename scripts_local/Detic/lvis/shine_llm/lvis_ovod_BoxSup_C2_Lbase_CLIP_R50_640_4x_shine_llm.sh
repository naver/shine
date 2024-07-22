#!/bin/bash

conda activate shine

METADATA_ROOT="nexus/lvis/shine_llm"

CUDA_VISIBLE_DEVICES=0 python train_net_detic.py \
        --num-gpus 1 \
        --config-file ./configs_detic/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.yaml\
        --eval-only \
        DATASETS.TEST "('lvis_v1_val',)" \
        MODEL.WEIGHTS ./models/detic/lvis_ovod/BoxSup-C2_Lbase_CLIP_R5021k_640b64_4x.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/lvis_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(1203,)" \
        MODEL.MASK_ON False
