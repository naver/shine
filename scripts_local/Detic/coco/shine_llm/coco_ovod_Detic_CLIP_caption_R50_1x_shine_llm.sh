#!/bin/bash

conda activate shine

METADATA_ROOT="nexus/coco/shine_llm"

python train_net_detic_coco.py \
        --num-gpus 1 \
        --config-file ./configs_detic/coco/Detic_OVCOCO_CLIP_R50_1x_caption.yaml \
        --eval-only \
        MODEL.WEIGHTS ./models/detic/coco_ovod/Detic_OVCOCO_CLIP_R50_1x_caption.pth \
        MODEL.RESET_CLS_TESTS True \
        MODEL.TEST_CLASSIFIERS "('${METADATA_ROOT}/coco_clip_hrchy_l1.npy',)" \
        MODEL.TEST_NUM_CLASSES "(80,)" \
        MODEL.MASK_ON False