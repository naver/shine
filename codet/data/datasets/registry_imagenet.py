from detectron2.data import DatasetCatalog, MetadataCatalog
from .registry_lvis_v1 import custom_load_lvis_json

__all__ = ["custom_register_imagenet_instances"]


def custom_register_imagenet_instances(name, metadata, json_file, image_root):
    """
    """
    DatasetCatalog.register(
        name,
        lambda: custom_load_lvis_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(
        json_file=json_file, image_root=image_root,
        evaluator_type="imagenet", **metadata
    )
