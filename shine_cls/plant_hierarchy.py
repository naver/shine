from tqdm import tqdm
import random
import os
from utils.fileios import *
from scipy.special import softmax
from PIL import Image
from collections import defaultdict
from utils.composer import SignatureComposer
from data_utils.cnames_imagenet import IMAGENET_CLASSES
import argparse



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='cluster', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--source',
                        type=str,
                        default="wordnet",
                        choices=["wordnet", "chatgpt", "chatgpt_post", "breeds"]
                        )


    args = parser.parse_args()

    SOURCE = args.source

    if SOURCE in ["wordnet"]:
        import tensorflow_datasets as tfds
        import tensorflow as tf
        from data_utils.cnames_imagenet import IMAGENET_CLASSES, MAPPER_WDNID


        output_path = "./hrchy_imagenet1k/imagenet1k_hrchy_wordnet.json"

        words_map = {}
        child_map = {}
        parent_map = {}
        gloss_map = {}

        words_path = os.path.join("./imagenet-ancestors-descendants", "words.txt")
        gloss_path = os.path.join("./imagenet-ancestors-descendants", "gloss.txt")
        child_map_path = os.path.join("./imagenet-ancestors-descendants", "is_a.txt")
        imagenet_label_to_wordnet_file = os.path.join("./imagenet-ancestors-descendants",
                                                      "imagenet_label_to_wordnet_synset.txt")

        blank = ' '
        comma_blank = ', '

        # obtain wordnet_id mappings for all words
        with tf.io.gfile.GFile(words_path, mode='r') as f:
            for line in f:
                line_split = line.split()
                wnid = line_split[0]
                words = line_split[1:]
                words_map[wnid] = words
        f.close()

        # obtain wordnet_id mappings for all word description
        with tf.io.gfile.GFile(gloss_path, mode='r') as f:
            for line in f:
                line_split = line.split()
                wnid = line_split[0]
                gloss = blank.join(line_split[1:])
                gloss_map[wnid] = gloss
        f.close()

        # obtain wordnet_id mappings for all parents-children
        with tf.io.gfile.GFile(child_map_path, mode='r') as f:
            for line in f:
                parent, child = line.split()
                parent_map[child] = parent
                if parent not in child_map:
                    child_map[parent] = [child]
                else:
                    child_map[parent].append(child)
        f.close()


        def get_samewords(wnid):
            return words_map[wnid]


        def get_parents(wnid):
            ancestors = []
            if wnid in parent_map:
                node = parent_map[wnid]  # only one parent class
            else:
                node = wnid
            while node in parent_map:  # one way go up
                ancestors.append(blank.join(words_map[node]))  # keep all ancestor
                node = parent_map[node]
            return ancestors


        def get_childs(wnid):
            descendants = []
            search = False
            if wnid in child_map:
                search = [child for child in child_map[wnid]]
            while search:  # go over all children (BFS)
                node = search.pop()
                descendants.append(blank.join(words_map[node]))  # keep all descendant
                if node in child_map:  # has children
                    [search.append(child) for child in child_map[node]]
            return descendants


        def get_one_parents(wnid):
            ancestors = []
            if wnid in parent_map:
                node = parent_map[wnid]  # only one parent class
            else:
                node = wnid
            while node in parent_map and len(ancestors) == 0:  # one way go up
                ancestors.append(blank.join(words_map[node]))  # keep all ancestor
                node = parent_map[node]
            return ancestors


        def get_one_childs(wnid):
            descendants = []
            search = False
            if wnid in child_map:
                search = [child for child in child_map[wnid]]
            while search:  # go over all children (BFS)
                node = search.pop()
                descendants.append(blank.join(words_map[node]))  # keep all descendant
                # if node in child_map:  # has children
                #     [search.append(child) for child in child_map[node]]
            return descendants


        def imagenet2wordnet():
            index_wdid = {}
            with tf.io.gfile.GFile(imagenet_label_to_wordnet_file, mode='r') as f:
                for line in f:
                    if "{'id'" in line:
                        index_wdid[line.split(": {'")[0].split("{")[-1].split(" ")[-1]] = 'n' + \
                                                                                          line.split("-n")[0].split(
                                                                                              "'")[
                                                                                              -1]
            return index_wdid

        isa_composer = SignatureComposer(prompter='isa')

        bad_words = ["substance", "matter", "object", "physical object", "physical entity"]

        hierarchy_tree = defaultdict(dict)
        for label_id, cname in enumerate(tqdm(IMAGENET_CLASSES)):
            wnid = MAPPER_WDNID[str(label_id)]
            raw_same_names, raw_parent_names, raw_child_names = [], [], []
            same_names, parent_names, child_names = [], [], []

            raw_same_names = get_samewords(wnid)
            raw_same_names = [cname]

            raw_parent_names = get_one_parents(wnid)
            raw_child_names = get_one_childs(wnid)


            same_names = [cname.replace(",", "") for cname in raw_same_names]

            for par_name in raw_parent_names:
                split_names = par_name.split(",")
                split_names = [name.strip() for name in split_names if name != ""]
                parent_names.extend(split_names)

            for child_name in raw_child_names:
                split_names = child_name.split(",")
                split_names = [name.strip() for name in split_names if name != ""]
                child_names.extend(split_names)

            same_names = [name for name in same_names if name not in bad_words]
            parent_names = [name for name in parent_names if name not in bad_words]
            child_names = [name for name in child_names if name not in bad_words]

            if len(child_names) == 0:
                signature_names = [
                    [my_name, parent_name]
                    for parent_name in parent_names
                    for my_name in same_names
                ]
            else:
                signature_names = [
                    [child_name, my_name, parent_name]
                    for parent_name in parent_names
                    for my_name in same_names
                    for child_name in child_names
                ]

            candidate_sentences = isa_composer.compose(signature_names)

            hierarchy_tree[str(label_id)] = {
                "node_name": cname,
                "same_names": same_names,
                "parent_names": parent_names,
                "child_names": child_names,
                "candidate_sentences": candidate_sentences,
            }

        dump_json(output_path, hierarchy_tree)
    elif SOURCE == "chatgpt":
        from data_utils.cnames_imagenet import IMAGENET_CLASSES, MAPPER_WDNID
        from utils.llm_controllers import LLMBot
        from utils.composer import SignatureComposer
        from utils.hierarchy_generator import HrchyPrompter

        output_path = "./hrchy_imagenet1k/imagenet1k_hrchy_llm.json"
        isa_composer = SignatureComposer(prompter='isa')

        context = ['types', 'object']

        results_dict = {str(cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': []}
                        for cat_id, cat_name in enumerate(IMAGENET_CLASSES)}

        CHATGPT_ZOO = ['gpt-3.5-turbo']

        bot = LLMBot(CHATGPT_ZOO[0])

        h_prompter = HrchyPrompter(dataset_name="lvis", num_sub=10, num_super=3)

        query_times = 3
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'], context=context)

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"[{cat_id}] Question A: {ppt_childs}")
            for i in range(query_times):
                print(f"Answer A-{1 + i}: {child_answers[i]}")

            print(f"[{cat_id}] Question B: {ppt_parents}")
            for i in range(query_times):
                print(f"Answer B-{1 + i}: {parent_answers[i]}")
            print('\n')
            # if int(cat_id) >= 3:
            #     break

        dump_json(output_path, results_dict)
    elif SOURCE == "chatgpt_post":
        input_path = "./hrchy_imagenet1k/imagenet1k_hierarchy_chatgpt.json"
        output_path = "./hrchy_imagenet1k/imagenet1k_hierarchy_chatgpt_composed.json"
        isa_composer = SignatureComposer(prompter='isa')
        bad_words = ["substance", "matter", "object", "physical object", "physical entity"]


        hierarchy_tree = defaultdict(dict)
        gpt_results = load_json(input_path)
        for label_id, entry in tqdm(sorted(gpt_results.items(), key=lambda item: int(item[0]))):
            raw_parents = {
                name.strip()
                for dirty_parent in entry["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100 and name.strip().lower() not in bad_words
            }

            raw_childs = {
                name.strip()
                for dirty_child in entry["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100 and name.strip().lower() not in bad_words
            }

            parent_names, child_names = list(raw_parents), list(raw_childs)

            if len(child_names) == 0:
                signature_names = [
                    [entry["node_name"], parent_name]
                    for parent_name in parent_names
                ]
            elif len(parent_names) == 0:
                signature_names = [
                    [child_name, entry["node_name"]]
                    for child_name in child_names
                ]
            else:
                signature_names = [
                    [child_name, entry["node_name"], parent_name]
                    for parent_name in parent_names
                    for child_name in child_names
                ]

            candidate_sentences = isa_composer.compose(signature_names)

            hierarchy_tree[str(label_id)] = {
                "node_name": entry["node_name"],
                "same_names": [entry["node_name"]],
                "parent_names": parent_names,
                "child_names": child_names,
                "candidate_sentences": candidate_sentences,
            }

        dump_json(output_path, hierarchy_tree)
    elif SOURCE == "breeds":
        def compose_hierarchy(in_result):
            bad_words = ["substance", "matter", "object", "physical object", "physical entity", "entity"]
            isa_composer = SignatureComposer(prompter='isa')

            out_result = defaultdict(dict)
            out_mapper = defaultdict()
            for cat_id, entry in tqdm(sorted(in_result.items(), key=lambda item: int(item[0]))):
                # update mapper_labelID
                if entry["node_id_at_leaf"] >= 0:
                    out_mapper[str(entry["node_id_at_leaf"])] = int(cat_id)
                else:
                    for leafID in entry["child_labels"]:
                        out_mapper[str(leafID)] = int(cat_id)

                # create clean entries
                node_name = entry["node_name"]
                parent_names = []
                child_names = []

                for par_name in entry["parent_names"]:
                    split_par_names = par_name.split(",")
                    split_par_names = [name.strip() for name in split_par_names if name != ""]
                    parent_names.extend(split_par_names)

                for chi_name in entry["child_names"]:
                    split_chi_names = chi_name.split(",")
                    split_chi_names = [name.strip() for name in split_chi_names if name != ""]
                    child_names.extend(split_chi_names)

                parent_names = [name for name in parent_names if name not in bad_words]
                child_names = [name for name in child_names if name not in bad_words]

                if len(child_names) == 0:
                    signature_names = [
                        [node_name, parent_name]
                        for parent_name in parent_names
                    ]
                elif len(parent_names) == 0:
                    signature_names = [
                        [child_name, node_name]
                        for child_name in child_names
                    ]
                else:
                    signature_names = [
                        [child_name, node_name, parent_name]
                        for parent_name in parent_names
                        for child_name in child_names
                    ]

                candidate_sentences = isa_composer.compose(signature_names)

                out_result[str(cat_id)] = {
                    "node_name": node_name,
                    "parent_names": parent_names,
                    "child_names": child_names,
                    "candidate_sentences": candidate_sentences,
                }
            # sort the output based on label IDs
            out_result.update(sorted(out_result.items(), key=lambda item: int(item[0])))
            out_mapper.update(sorted(out_mapper.items(), key=lambda item: int(item[0])))
            return out_result, out_mapper


        hierarchy_root = "hrchy_breeds"

        input_paths = {
            'l1': f"{hierarchy_root}/breed_l1_num_class=2.json",
            'l2': f"{hierarchy_root}/breed_l2_num_class=10.json",
            'l3': f"{hierarchy_root}/breed_l3_num_class=29.json",
            'l4': f"{hierarchy_root}/breed_l4_num_class=128.json",
            'l5': f"{hierarchy_root}/breed_l5_num_class=466.json",
            'l6': f"{hierarchy_root}/breed_l6_num_class=591.json",
            'l7': f"{hierarchy_root}/breed_l7_num_class=98.json",
        }
        output_result_paths = {
            'l1': f"{hierarchy_root}/composed_breed_l1_num_class=2.json",
            'l2': f"{hierarchy_root}/composed_breed_l2_num_class=10.json",
            'l3': f"{hierarchy_root}/composed_breed_l3_num_class=29.json",
            'l4': f"{hierarchy_root}/composed_breed_l4_num_class=128.json",
            'l5': f"{hierarchy_root}/composed_breed_l5_num_class=466.json",
            'l6': f"{hierarchy_root}/composed_breed_l6_num_class=591.json",
            'l7': f"{hierarchy_root}/composed_breed_l7_num_class=98.json",
        }
        output_mapper_paths = {
            'l1': f"{hierarchy_root}/mapper_l1_leaf2current.json",
            'l2': f"{hierarchy_root}/mapper_l2_leaf2current.json",
            'l3': f"{hierarchy_root}/mapper_l3_leaf2current.json",
            'l4': f"{hierarchy_root}/mapper_l4_leaf2current.json",
            'l5': f"{hierarchy_root}/mapper_l5_leaf2current.json",
            'l6': f"{hierarchy_root}/mapper_l6_leaf2current.json",
            'l7': f"{hierarchy_root}/mapper_l7_leaf2current.json",
        }

        for k_lname, v_in_path in input_paths.items():
            print(f"Compose the hierarchy for {k_lname}...")
            in_result = load_json(v_in_path)
            out_result, out_mapper = compose_hierarchy(in_result)

            dump_json(output_result_paths[k_lname], out_result)
            print(f"Succ. dumped {k_lname} result to: " + output_result_paths[k_lname])

            dump_json(output_mapper_paths[k_lname], out_mapper)
            print(f"Succ. dumped {k_lname} mapper to: " + output_mapper_paths[k_lname])




































