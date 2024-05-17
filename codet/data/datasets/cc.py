# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/facebookresearch/Detic/blob/main/detic/data/datasets/cc.py
# Modified by Mingxuan Liu

import logging
import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .registry_lvis_v1 import custom_register_lvis_instances

_CUSTOM_SPLITS = {
    "cc3m_v1_val": ("cc3m/validation/", "cc3m/val_image_info.json"),
    "cc3m_v1_train": ("cc3m/training/", "cc3m/train_image_info.json"),
    "cc3m_v1_train_tags": ("cc3m/training/", "cc3m/train_image_info_tags.json"),
}


def register_all_custom_cc(root="datasets"):
    for key, (image_root, json_file) in _CUSTOM_SPLITS.items():
        custom_register_lvis_instances(
            key,
            get_lvis_instances_meta('lvis_v1'),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# if __name__.endswith(".builtin_cc"):
#     # Assume pre-defined datasets live in `./datasets`.
#     _root = "datasets"
#     register_all_custom_cc(_root)
