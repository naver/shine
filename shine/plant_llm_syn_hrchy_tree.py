import argparse
import json
import torch
import numpy as np
import itertools
from nltk.corpus import wordnet
import sys
import clip
import os
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import re
from copy import deepcopy
from collections import defaultdict

import ast
from shine.tools.llm_controllers import LLMBot, HrchyPrompter
from shine.tools.composer import SignatureComposer
from shine.tools.fileios import *

def compose_isa(signature_list):
    return ['a ' + signature[0] + ''.join([f', which is a {parentName}' for parentName in signature[1:]])
            for signature in signature_list]


def try_deserialize_ast(input_str):
    try:
        return ast.literal_eval(input_str)
    except (ValueError, SyntaxError):
        return [input_str]


def remove_symbols(messy_sent):
    messy_sent = str(messy_sent)
    messy_sent = messy_sent.replace('[', '')
    messy_sent = messy_sent.replace(']', '')
    messy_sent = messy_sent.strip()

    for _ in range(3):
        if messy_sent.startswith("'") or messy_sent.startswith(",") or messy_sent.startswith("."):
            messy_sent = messy_sent[1:].strip()
    for _ in range(3):
        if messy_sent.endswith("'") or messy_sent.endswith(",") or messy_sent.endswith("."):
            messy_sent = messy_sent[:-1].strip()
    return messy_sent.strip()


def organize_sentences(sentences):
    if '\n' in sentences:
        raw_reply = sentences.split('\n')
    else:
        raw_replay = remove_symbols(sentences)
        raw_reply = raw_replay.split("',")

    cleaned_reply = [remove_symbols(raw_sent) for raw_sent in raw_reply]
    return cleaned_reply


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod', 'oid_lvis', 'lvis',
                                                                                'oid', 'coco'])
    parser.add_argument('--mode', default='postprocess', choices=['query', 'postprocess'])
    parser.add_argument('--output_root', default='')
    parser.add_argument('--h_level', default='l1', choices=['l6', 'l5', 'l4', 'l3', 'l2', 'l1'])
    parser.add_argument('--num_sub', type=int, default=10)
    parser.add_argument('--num_super', type=int, default=3)
    parser.add_argument('--query_times', type=int, default=3)

    args = parser.parse_args()

    if not is_valid_folder(args.output_root): raise FileExistsError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    CHATGPT_ZOO = ['gpt-3.5-turbo']

    bot = LLMBot(CHATGPT_ZOO[0])

    args.output_path = os.path.join(args.output_root, f"raw_{args.dataset_name}_gpt_hrchy_{args.h_level}")

    if args.mode == 'query' and args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        level_contexts = {
            'l6': ['types', 'Species'],
            'l5': ['Specie', 'Genus'],
            'l4': ['Genus', 'Family'],
            'l3': ['Family', 'Order'],
            'l2': ['Orders', 'Class'],
            'l1': ['Class', 'Phylum']
        }
        starting_idx = level_names.index(args.h_level)
        level_names = level_names[starting_idx:]

        print('Loading hierarchy tree from inat_annotations/inat_hierarchy_tree.json')
        meta_tree = json.load(open('inat_annotations/inat_hierarchy_tree.json', 'r'))
        meta_level = meta_tree.get(args.h_level)

        if args.h_level == 'l1':
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            signature_names = [[x['name']] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [[] for sig_name in signature_names]
        else:
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            for i in range(len(signature_ids)):
                leaf_id = str(signature_ids[i][0])
                parents_ids = meta_level['parents'].get(leaf_id)
                signature_ids[i].extend(parents_ids)

            signature_names = []
            for cat_id in signature_ids:
                cat_name = []
                for level_idx, this_id in enumerate(cat_id):
                    level_name = level_names[level_idx]
                    this_name = meta_tree[level_name]['categories'][this_id - 1]['name']
                    cat_name.append(this_name)
                signature_names.append(cat_name)

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [sig_name[1:] for sig_name in signature_names]

        results_dict = {cat_id: {'node_name': cat_name, 'parent_names': cat_parents, 'child_names': []}
                        for cat_id, cat_name, cat_parents in zip(node_ids, node_names, node_parents)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt = h_prompter.embed(node_name=cat_entry['node_name'], context=level_contexts[args.h_level])
            answers = [bot.infer(ppt, temperature=0.7) for i in range(args.query_times)]
            results_dict[cat_id]['child_names'] = answers
            print(f"Question: {ppt}")
            for i in range(args.query_times):
                print(f"Answer {1+i}: {answers[i]}")
            print('\n')

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'fsod':
        level_names = ['l3', 'l2', 'l1']
        level_contexts = {
            'l3': ['types', 'object'],
            'l2': ['types', 'object'],
            'l1': ['types', 'object'],
        }
        starting_idx = level_names.index(args.h_level)
        level_names = level_names[starting_idx:]

        print('Loading hierarchy tree from fsod_annotations/fsod_hierarchy_tree.json')
        meta_tree = json.load(open('fsod_annotations/fsod_hierarchy_tree.json', 'r'))
        meta_level = meta_tree.get(args.h_level)

        if args.h_level == 'l1':
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            signature_names = [[x['name']] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [[] for sig_name in signature_names]
        else:
            signature_ids = [[int(x['id'])] for x in sorted(meta_level['categories'], key=lambda x: x['id'])]
            for i in range(len(signature_ids)):
                leaf_id = str(signature_ids[i][0])
                parents_ids = meta_level['parents'].get(leaf_id)
                signature_ids[i].extend(parents_ids)

            signature_names = []
            for cat_id in signature_ids:
                cat_name = []
                for level_idx, this_id in enumerate(cat_id):
                    level_name = level_names[level_idx]
                    this_name = meta_tree[level_name]['categories'][this_id - 1]['name']
                    cat_name.append(this_name)
                signature_names.append(cat_name)

            node_ids = [str(sig_id[0]) for sig_id in signature_ids]
            node_names = [sig_name[0] for sig_name in signature_names]
            node_parents = [sig_name[1:] for sig_name in signature_names]

        results_dict = {cat_id: {'node_name': cat_name, 'parent_names': cat_parents, 'child_names': []}
                        for cat_id, cat_name, cat_parents in zip(node_ids, node_names, node_parents)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'],
                                                       context=level_contexts[args.h_level])

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(args.query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(args.query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"Question A: {ppt_childs}")
            for i in range(args.query_times):
                print(f"Answer A-{1+i}: {child_answers[i]}")

            print(f"Question B: {ppt_parents}")
            for i in range(args.query_times):
                print(f"Answer B-{1+i}: {parent_answers[i]}")
            print('\n')

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'oid_lvis':
        from oid_annotations.class_names import categories_all

        lvis_meta = load_json("lvis_annotations/lvis_v1_train_cat_info.json")
        oid_meta = categories_all

        lvis_cnames = set([entry["name"].replace("_", " ").replace("-", " ").lower() for entry in lvis_meta])
        oid_cnames = set([entry['name'].replace("_", " ").replace("-", " ").lower() for entry in oid_meta])

        combined_cnames = lvis_cnames.union(oid_cnames)

        print(f"{len(combined_cnames)} (combined_cnames) = {len(lvis_cnames)} (lvis) + {len(oid_cnames)} (oid)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'], context=context)

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(args.query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(args.query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"[{cat_id}] Question A: {ppt_childs}")
            for i in range(args.query_times):
                print(f"Answer A-{1 + i}: {child_answers[i]}")

            print(f"[{cat_id}] Question B: {ppt_parents}")
            for i in range(args.query_times):
                print(f"Answer B-{1 + i}: {parent_answers[i]}")
            print('\n')
            # if int(cat_id) >= 3:
            #     break

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'oid':
        from oid_annotations.class_names import categories_all

        oid_meta = categories_all
        oid_cnames = [entry['name'].replace("_", " ").replace("-", " ").lower() for entry in oid_meta]
        combined_cnames = oid_cnames

        print(f"{len(combined_cnames)} (combined_cnames) = {len(oid_cnames)} (oid)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'], context=context)

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(args.query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(args.query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"[{cat_id}] Question A: {ppt_childs}")
            for i in range(args.query_times):
                print(f"Answer A-{1 + i}: {child_answers[i]}")

            print(f"[{cat_id}] Question B: {ppt_parents}")
            for i in range(args.query_times):
                print(f"Answer B-{1 + i}: {parent_answers[i]}")
            print('\n')
            # if int(cat_id) >= 3:
            #     break

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'lvis':
        from oid_annotations.class_names import categories_all

        lvis_meta = load_json("lvis_annotations/lvis_v1_train_cat_info.json")
        lvis_cnames = [entry["name"].replace("_", " ").replace("-", " ").lower() for entry in lvis_meta]
        combined_cnames = lvis_cnames

        oid_lvis_results = load_json("openset_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_hrchy_l1.json")

        print(f"{len(combined_cnames)} (combined_cnames) = {len(lvis_cnames)} (lvis)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': [],
                                        'candidate_sentences': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}

        def retrieve_oid_lvis(node_name):
            for key, value in list(oid_lvis_results.items()):
                if value['node_name'] == node_name:
                    return deepcopy(value)

        for cat_id, cat_entry in list(results_dict.items()):
            results_dict[cat_id] = retrieve_oid_lvis(cat_entry['node_name'])

        dump_json(args.output_path, results_dict)
    elif args.mode == 'query' and args.dataset_name == 'coco':
        coco_meta = load_json("coco_annotations/instances_val2017_all_2_oriorder_cat_info.json")
        coco_cnames = [entry["name"].replace("_", " ").replace("-", " ").lower() for entry in coco_meta]
        combined_cnames = coco_cnames

        print(f"{len(combined_cnames)} (combined_cnames) = {len(coco_cnames)} (coco)")

        context = ['types', 'object']

        results_dict = {str(1+cat_id): {'node_name': cat_name, 'parent_names': [], 'child_names': []}
                        for cat_id, cat_name in enumerate(combined_cnames)}

        h_prompter = HrchyPrompter(dataset_name=args.dataset_name, num_sub=args.num_sub, num_super=args.num_super)
        for cat_id, cat_entry in results_dict.items():
            ppt_childs, ppt_parents = h_prompter.embed(node_name=cat_entry['node_name'], context=context)

            child_answers = [bot.infer(ppt_childs, temperature=0.7) for i in range(args.query_times)]
            parent_answers = [bot.infer(ppt_parents, temperature=0.7) for i in range(args.query_times)]

            results_dict[cat_id]['child_names'] = child_answers
            results_dict[cat_id]['parent_names'] = parent_answers

            print(f"[{cat_id}] Question A: {ppt_childs}")
            for i in range(args.query_times):
                print(f"Answer A-{1 + i}: {child_answers[i]}")

            print(f"[{cat_id}] Question B: {ppt_parents}")
            for i in range(args.query_times):
                print(f"Answer B-{1 + i}: {parent_answers[i]}")
            print('\n')
            # if int(cat_id) >= 3:
            #     break

        dump_json(args.output_path, results_dict)
    elif args.mode == 'postprocess' and args.dataset_name == 'inat':
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }
            v["child_names"] = list(clean_childs)

            parent_names = [v["parent_names"][0]] if v["parent_names"] else []

            signature_names = [
                [child_name, v["node_name"]] + parent_names
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    elif args.mode == 'postprocess' and args.dataset_name == 'fsod':
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_parents = {
                name.strip()
                for dirty_parent in v["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100
            }
            v["child_names"] = list(clean_childs)
            v["parent_names"] = list(clean_parents)

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"][:1]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned_'), raw_results)
    elif args.mode == 'postprocess' and args.dataset_name == 'oid_lvis':
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')
        oid_lvis_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                          for _, entry in raw_results.items()]

        fsod_results = load_json(f"fsod_llm_answers/r3_cleaned_fsod_gpt_hrchy_{args.h_level}")
        fsod_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                       for _, entry in fsod_results.items()]

        current_cnames = oid_lvis_cnames + fsod_cnames

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_parents = {
                name.strip()
                for dirty_parent in v["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_childs = list(clean_childs)
            clean_parents = list(clean_parents)

            trimmed_childs = [name for name in clean_childs
                              if name.replace("_", " ").replace("-", " ").lower() not in current_cnames]
            trimmed_parents = [name for name in clean_parents
                               if name.replace("_", " ").replace("-", " ").lower() not in current_cnames]

            v["child_names"] = trimmed_childs
            v["parent_names"] = trimmed_parents

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    elif args.mode == 'postprocess' and args.dataset_name in ['oid', 'coco']:
        raw_results = load_json(args.output_path)
        isa_composer = SignatureComposer(prompter='isa')

        current_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                          for _, entry in raw_results.items()]

        for k_cat, v in raw_results.items():
            clean_childs = {
                name.strip()
                for dirty_child in v["child_names"]
                for name in dirty_child.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_parents = {
                name.strip()
                for dirty_parent in v["parent_names"]
                for name in dirty_parent.split('&')
                if 3 <= len(name.strip()) <= 100
            }

            clean_childs = list(clean_childs)
            clean_parents = list(clean_parents)

            trimmed_childs = [name.lower() for name in clean_childs]
            trimmed_parents = [name.lower() for name in clean_parents]

            trimmed_childs = list(set(trimmed_childs))
            trimmed_parents = list(set(trimmed_parents))

            v["child_names"] = clean_childs
            v["parent_names"] = clean_parents

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    elif args.mode == 'postprocess' and args.dataset_name in ['lvis']:
        raw_results = load_json(args.output_path.replace('raw', 'cleaned'))
        isa_composer = SignatureComposer(prompter='isa')

        current_cnames = [entry["node_name"].replace("_", " ").replace("-", " ").lower()
                          for _, entry in raw_results.items()]

        for k_cat, v in raw_results.items():
            clean_childs = v["child_names"]
            clean_parents = v["parent_names"]

            trimmed_childs = [name.lower() for name in clean_childs]
            trimmed_parents = [name.lower() for name in clean_parents]

            trimmed_childs = list(set(trimmed_childs))
            trimmed_parents = list(set(trimmed_parents))

            v["child_names"] = clean_childs
            v["parent_names"] = clean_parents

            signature_names = [
                [child_name, v["node_name"], parent_name]
                for parent_name in v["parent_names"]
                for child_name in v["child_names"]
            ]

            v["candidate_sentences"] = isa_composer.compose(signature_names)

        dump_json(args.output_path.replace('raw', 'cleaned'), raw_results)
    else:
        raise NotImplementedError
