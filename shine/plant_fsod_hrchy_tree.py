from copy import deepcopy
from collections import defaultdict
from shine.tools.fileios import *

if __name__ == '__main__':
    per_layer_path = {
            "l1": "fsod_annotations/fsod_anno_fixed_l1.json",
            "l2": "fsod_annotations/fsod_anno_fixed_l2.json",
            "l3": "fsod_annotations/fsod_anno_fixed_l3.json",
    }

    per_layer_annos = {k: load_json(v) for k, v in per_layer_path.items()}

    level3 = {"categories": deepcopy(per_layer_annos['l3']['categories']), "childs": {}, "parents": {}}
    level2 = {"categories": deepcopy(per_layer_annos['l2']['categories']), "childs": {}, "parents": {}}
    level1 = {"categories": deepcopy(per_layer_annos['l1']['categories']), "childs": {}, "parents": {}}

    map_cname2id_l2 = {cat['name']: cat['id'] for cat in per_layer_annos['l2']['categories']}
    map_cname2id_l1 = {cat['name']: cat['id'] for cat in per_layer_annos['l1']['categories']}

    # Build L3
    #   |- build parents
    for this_cat in per_layer_annos['l3']["categories"]:
        level3["parents"][str(this_cat["id"])] = [map_cname2id_l2[this_cat["supercategory"]],
                                                  map_cname2id_l1[this_cat["supersupercategory"]]]

    # Build L2
    #   |- build parents
    for this_cat in per_layer_annos['l2']["categories"]:
        level2["parents"][str(this_cat["id"])] = [map_cname2id_l1[this_cat["supercategory"]]]
    #   |- build childs
    for this_cat in per_layer_annos['l2']["categories"]:
        this_name = this_cat['name']
        this_childs = []
        for l3_cat in per_layer_annos['l3']["categories"]:
            if l3_cat['supercategory'] == this_name:
                this_childs.append(l3_cat['id'])

        level2["childs"][str(this_cat["id"])] = this_childs

    # Build L1
    #   |- build childs
    for this_cat in per_layer_annos['l1']["categories"]:
        this_name = this_cat['name']
        this_childs = []
        for l2_cat in per_layer_annos['l2']["categories"]:
            if l2_cat['supercategory'] == this_name:
                this_childs.append(l2_cat['id'])

        level1["childs"][str(this_cat["id"])] = this_childs

    hierarchy_tree = {
        "l1": level1,
        "l2": level2,
        "l3": level3,
    }

    dump_json('fsod_annotations/fsod_hierarchy_tree.json', hierarchy_tree)




