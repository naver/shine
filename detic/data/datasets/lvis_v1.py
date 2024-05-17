# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/facebookresearch/Detic/blob/main/detic/data/datasets/lvis_v1.py
# Modified by Mingxuan Liu

import logging
import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .registry_lvis_v1 import custom_register_lvis_instances


def get_lvis_22k_meta():
    from .lvis_22k_categories import CATEGORIES
    cat_ids = [k["id"] for k in CATEGORIES]
    assert min(cat_ids) == 1 and max(cat_ids) == len(
        cat_ids
    ), "Category ids are not in [1, #categories], as expected"
    # Ensure that the category list is sorted by id
    lvis_categories = sorted(CATEGORIES, key=lambda x: x["id"])
    thing_classes = [k["name"] for k in lvis_categories]
    meta = {"thing_classes": thing_classes}
    return meta


_CUSTOM_SPLITS_LVIS = {
    "lvis_v1_train+coco": ("coco/", "lvis/lvis_v1_train+coco_mask.json"),
    "lvis_v1_train_norare": ("coco/", "lvis/lvis_v1_train_norare.json"),
}

_CUSTOM_SPLITS_LVIS_22K = {
    "lvis_v1_train_22k": ("coco/", "lvis/lvis_v1_train_lvis-22k.json"),
}


def register_all_custom_lvis(root="datasets"):
    for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS.items():
        custom_register_lvis_instances(
            key,
            get_lvis_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_custom_lvis_22k(root="datasets"):
    for key, (image_root, json_file) in _CUSTOM_SPLITS_LVIS_22K.items():
        custom_register_lvis_instances(
            key,
            get_lvis_22k_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# if __name__.endswith(".builtin_lvis_v1"):
#     # Assume pre-defined datasets live in `./datasets`.
#     _root = "datasets"
#     register_all_custom_lvis(_root)
#     register_all_custom_lvis_22k(_root)
