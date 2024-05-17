# Copyright (c) Facebook, Inc. and its affiliates.
# Part of the code is from https://github.com/facebookresearch/Detic/blob/main/detic/data/datasets/cc.py
# Modified by Mingxuan Liu

import logging
import os

from detectron2.data.datasets.lvis import get_lvis_instances_meta
from .lvis_v1 import get_lvis_22k_meta
from .registry_imagenet import custom_register_imagenet_instances


_CUSTOM_SPLITS_IMAGENET = {
    "imagenet_lvis_v1": ("imagenet/ImageNet-LVIS/", "imagenet/annotations/imagenet_lvis_image_info.json"),
}

_CUSTOM_SPLITS_IMAGENET_22K = {
    "imagenet_lvis-22k": ("imagenet/ImageNet-LVIS/", "imagenet/annotations/imagenet-22k_image_info_lvis-22k.json"),
}


def register_all_lvis_imagenet(root="datasets"):
    for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET.items():
        custom_register_imagenet_instances(
            key,
            get_lvis_instances_meta('lvis_v1'),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_lvis_imagenet22k(root="datasets"):
    for key, (image_root, json_file) in _CUSTOM_SPLITS_IMAGENET_22K.items():
        custom_register_imagenet_instances(
            key,
            get_lvis_22k_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


# if __name__.endswith(".builtin_imagenet"):
#     # Assume pre-defined datasets live in `./datasets`.
#     _root = "datasets"
#     register_all_lvis_imagenet(_root)
#     register_all_lvis_imagenet22k(_root)

