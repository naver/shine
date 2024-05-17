from .cc import register_all_custom_cc
from .coco_zeroshot import register_all_coco_zeroshot, register_all_coco_zeroshot_custom_split
from .imagenet import register_all_lvis_imagenet, register_all_lvis_imagenet22k
from .lvis_v1 import register_all_custom_lvis, register_all_custom_lvis_22k
from .objects365 import register_all_objects365
from .oid import register_all_oid, register_all_oid_hierarchy
from .inat import register_all_inat_hierarchy
from .fsod import register_all_fsod_hierarchy

_root = "datasets"


if __name__.endswith(".builtin"):
    # |- CC-COCO-captions
    register_all_custom_cc(_root)
    # |- COCO and COCO OpenVocabulary
    register_all_coco_zeroshot(_root)
    register_all_coco_zeroshot_custom_split(_root)
    # |- ImageNet
    register_all_lvis_imagenet(_root)
    register_all_lvis_imagenet22k(_root)
    # |- LVIS
    register_all_custom_lvis(_root)
    register_all_custom_lvis_22k(_root)
    # |- Objects365
    register_all_objects365(_root)
    # |- OpenImages
    register_all_oid(_root)
    register_all_oid_hierarchy(_root)
    # |- iNatLoc500 w/ 6-levels hierarchy
    register_all_inat_hierarchy(_root)
    # |- FSOD w/ 3-levels hierarchy
    register_all_fsod_hierarchy(_root)





