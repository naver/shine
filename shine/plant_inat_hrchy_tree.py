from collections import defaultdict
from inat_annotations.class_names import PER_LAYER_CATEGORY_NAMES
from shine.tools.fileios import *


def txt2imgId_pairs(path):
    pairs_dict = {}
    raw_t = load_txt(path)
    raw_t = raw_t.strip()
    raw_t = raw_t.split('\n')

    for lin in raw_t:
        imgId, catId = lin.split(',')[-2], lin.split(',')[-1]
        catId = int(catId) + 1
        pairs_dict[imgId] = catId
    # img_path: cat_id
    return pairs_dict


def generate_hierarchy_tree(list_signature):
    catId_parents = [defaultdict(list) for _ in range(6)]
    catId_childs  = [defaultdict(list) for _ in range(6)]

    for signature in list_signature:
        # Using loop to avoid repetition
        for idx in range(6):
            # For parents dictionary
            if idx < 5:
                catId_parents[idx][signature[idx]] = signature[idx + 1:]    # all the parents

            # For child dictionary
            if idx > 0:
                sub_cat, super_cat = signature[idx - 1], signature[idx]
                # Append sub_cat if it's not already in the list
                if sub_cat not in catId_childs[idx][super_cat]:
                    catId_childs[idx][super_cat].append(sub_cat)

    hierarchy_tree = {}
    for idx, level in enumerate(['l6', 'l5', 'l4', 'l3', 'l2', 'l1']):
        hierarchy_tree[level] = {
            "categories": PER_LAYER_CATEGORY_NAMES[level],
            "childs": dict(catId_childs[idx]),  # Convert back to regular dict
            "parents": dict(catId_parents[idx])  # Convert back to regular dict
        }
    return hierarchy_tree


if __name__ == '__main__':
    per_layer_path = {
            # "l0": "hierarchy/class_names_kingdom.txt",
            "l1": "inat_annotations/class_labels_phylum.txt",
            "l2": "inat_annotations/class_labels_class.txt",
            "l3": "inat_annotations/class_labels_order.txt",
            "l4": "inat_annotations/class_labels_family.txt",
            "l5": "inat_annotations/class_labels_genus.txt",
            "l6": "inat_annotations/class_labels.txt",
    }

    per_layer_pairs = {k: txt2imgId_pairs(v) for k, v in per_layer_path.items()}

    anno_tree_from_leaf = defaultdict(list)     # img_id: [cat_id_L6, ..., cat_id_L1]
    order_leaf2root = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']

    for L in order_leaf2root:
        for imgId, catId in per_layer_pairs[L].items():
            anno_tree_from_leaf[imgId].append(catId)

    list_signature = list(anno_tree_from_leaf.values())

    hierarchy_tree = generate_hierarchy_tree(list_signature)
    dump_json('inat_annotations/inat_hierarchy_tree.json', hierarchy_tree)




