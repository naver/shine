import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from copy import deepcopy
from shine.tools.themer import Themer
from shine.tools.fileios import *
from torch.nn import functional as F


coco_novel = [
    'airplane',
    'bus',
    'cat',
    'dog',
    'cow',
    'elephant',
    'umbrella',
    'tie',
    'snowboard',
    'skateboard',
    'cup',
    'knife',
    'cake',
    'couch',
    'keyboard',
    'sink',
    'scissors',
]

lvis_novel = ['applesauce', 'apricot', 'arctic (type of shoe)', 'armoire', 'armor', 'ax', 'baboon', 'bagpipe', 'baguet', 'bait', 'ballet skirt', 'banjo', 'barbell', 'barge', 'bass horn', 'batter (food)', 'beachball', 'bedpan', 'beeper', 'beetle', 'bible', 'birthday card', 'pirate flag', 'blimp', 'gameboard', 'bob', 'bolo tie', 'bonnet', 'bookmark', 'boom microphone', 'bow (weapon)', 'pipe bowl', 'bowling ball', 'boxing glove', 'brass plaque', 'breechcloth', 'broach', 'bubble gum', 'horse buggy', 'bulldozer', 'bulletproof vest', 'burrito', 'cabana', 'locker', 'candy bar', 'canteen', 'elevator car', 'car battery', 'cargo ship', 'carnation', 'casserole', 'cassette', 'chain mail', 'chaise longue', 'chalice', 'chap', 'checkbook', 'checkerboard', 'chessboard', 'chime', 'chinaware', 'poker chip', 'chocolate milk', 'chocolate mousse', 'cider', 'cigar box', 'clarinet', 'cleat (for securing rope)', 'clementine', 'clippers (for plants)', 'cloak', 'clutch bag', 'cockroach', 'cocoa (beverage)', 'coil', 'coloring material', 'combination lock', 'comic book', 'compass', 'convertible (automobile)', 'sofa bed', 'cooker', 'cooking utensil', 'corkboard', 'cornbread', 'cornmeal', 'cougar', 'coverall', 'crabmeat', 'crape', 'cream pitcher', 'crouton', 'crowbar', 'hair curler', 'curling iron', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'date (fruit)', 'detergent', 'diary', 'die', 'dinghy', 'tux', 'dishwasher detergent', 'diving board', 'dollar', 'dollhouse', 'dove', 'dragonfly', 'drone', 'dropper', 'drumstick', 'dumbbell', 'dustpan', 'earplug', 'eclair', 'eel', 'egg roll', 'electric chair', 'escargot', 'eyepatch', 'falcon', 'fedora', 'ferret', 'fig (fruit)', 'file (tool)', 'first aid kit', 'fishbowl', 'flash', 'fleece', 'football helmet', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'gargoyle', 'gasmask', 'gemstone', 'generator', 'goldfish', 'gondola (boat)', 'gorilla', 'gourd', 'gravy boat', 'griddle', 'grits', 'halter top', 'hamper', 'hand glass', 'handcuff', 'handsaw', 'hardback book', 'harmonium', 'hatbox', 'headset', 'heron', 'hippopotamus', 'hockey stick', 'hookah', 'hornet', 'hot air balloon', 'hotplate', 'hourglass', 'houseboat', 'hummus', 'popsicle', 'ice pack', 'ice skate', 'inhaler', 'jelly bean', 'jewel', 'joystick', 'keg', 'kennel', 'keycard', 'kitchen table', 'knitting needle', 'knocker (on a door)', 'koala', 'lab coat', 'lamb chop', 'lasagna', 'lawn mower', 'leather', 'legume', 'lemonade', 'lightning rod', 'limousine', 'liquor', 'machine gun', 'mallard', 'mallet', 'mammoth', 'manatee', 'martini', 'mascot', 'masher', 'matchbox', 'microscope', 'milestone', 'milk can', 'milkshake', 'mint candy', 'motor vehicle', 'music stool', 'nailfile', 'neckerchief', 'nosebag (for animals)', 'nutcracker', 'octopus (food)', 'octopus (animal)', 'omelet', 'inkpad', 'pan (metal container)', 'pantyhose', 'papaya', 'paperback book', 'paperweight', 'parchment', 'passenger ship', 'patty (food)', 'wooden leg', 'pegboard', 'pencil box', 'pencil sharpener', 'pendulum', 'pennant', 'penny (coin)', 'persimmon', 'phonebook', 'piggy bank', 'pin (non jewelry)', 'ping pong ball', 'pinwheel', 'tobacco pipe', 'pistol', 'pitchfork', 'playpen', 'plow (farm equipment)', 'plume', 'pocket watch', 'poncho', 'pool table', 'prune', 'pudding', 'puffer (fish)', 'puffin', 'pug dog', 'puncher', 'puppet', 'quesadilla', 'quiche', 'race car', 'radar', 'rag doll', 'rat', 'rib (food)', 'river boat', 'road map', 'rodent', 'roller skate', 'rollerblade', 'root beer', 'safety pin', 'salad plate', 'salmon (food)', 'satchel', 'saucepan', 'sawhorse', 'saxophone', 'scarecrow', 'scraper', 'seaplane', 'sharpener', 'sharpie', 'shaver (electric)', 'shawl', 'shears', 'shepherd dog', 'sherbert', 'shot glass', 'shower cap', 'shredder (for paper)', 'skullcap', 'sling (bandage)', 'smoothie', 'snake', 'softball', 'sombrero', 'soup bowl', 'soya milk', 'space shuttle', 'sparkler (fireworks)', 'spear', 'crawfish', 'squid (food)', 'stagecoach', 'steak knife', 'stepladder', 'stew', 'stirrer', 'string cheese', 'stylus', 'subwoofer', 'sugar bowl', 'sugarcane (plant)', 'syringe', 'tabasco sauce', 'table tennis table', 'tachometer', 'taco', 'tambourine', 'army tank', 'telephoto lens', 'tequila', 'thimble', 'trampoline', 'trench coat', 'triangle (musical instrument)', 'truffle (chocolate)', 'vat', 'turnip', 'unicycle', 'vinegar', 'violin', 'vodka', 'vulture', 'waffle iron', 'walrus', 'wardrobe', 'washbasin', 'water heater', 'water gun', 'wolf']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat_expanded', choices=['inat_expanded',
                                                                                         'fsod_expanded',
                                                                                         'coco', 'lvis', 'oid'])
    parser.add_argument('--gpt_results_root', default='inat_llm_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens',
                                                                              'plain'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--clip_model', default="ViT-B/32")
    parser.add_argument('--out_path', default='')

    args = parser.parse_args()

    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name in ["inat_expanded", "fsod_expanded"]:
        if args.dataset_name == 'inat_expanded':
            args.dataset_name = 'inat'
            level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
        else:
            args.dataset_name = 'fsod'
            level_names = ['l3', 'l2', 'l1']

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

        theme_maker = Themer(method=args.aggregator if args.aggregator != "plain" else "mean",
                             thresh=args.peigen_thresh,
                             alpha=args.alpha)

        theme_tree_features = defaultdict(dict)
        for level_name in level_names[:1]:
            gpt_results = load_json(os.path.join(args.gpt_results_root,
                                                 f"cleaned_{args.dataset_name}_gpt_hrchy_{level_name}.json"))

            expanded_results = load_json(f"miss_lvis_oid_llm_answers/cleaned_oid_lvis_gpt_hrchy_{level_name}.json")

            # Removing overlapping node_names from B
            overlapping_names = {entry['node_name'].replace("_", " ").replace("-", " ").lower()
                                 for entry in gpt_results.values()}

            for key, value in list(expanded_results.items()):  # using list to create a copy so we can modify the expanded results in-place
                if value['node_name'] in overlapping_names:
                    del expanded_results[key]

            # Merging dictionaries with updated keys for the expanded categories
            next_key = len(gpt_results) + 1
            merged_dict = deepcopy(gpt_results)  # start with a copy of the target categories

            for value in expanded_results.values():
                merged_dict[str(next_key)] = value
                next_key += 1

            for cat_id, entry in sorted(merged_dict.items(), key=lambda item: int(item[0])):
                node_sentences = entry["candidate_sentences"] if args.aggregator != "plain" else entry["node_name"]
                node_tokens = clip.tokenize(node_sentences).to(device)
                with torch.no_grad():
                    node_features = global_encoder.encode_text(node_tokens)
                node_features = F.normalize(node_features)
                # if node_features.size(0) == 1:
                #     print(f"level={level_name} cat={cat_id}")
                #     print(node_sentences)
                #     sys.exit()
                node_theme = theme_maker.get_theme(node_features)
                theme_tree_features[level_name][cat_id] = node_theme

        for level_name, level_ids in theme_tree_features.items():
            total_num = len(list(level_ids.values()))
            print(f"Total feats = {total_num} at {level_name}")

        # Prepare and Save Features
        for level_name, level_theme_dict in theme_tree_features.items():
            sorted_theme_dict = OrderedDict(sorted(level_theme_dict.items(), key=lambda x: int(x[0])))

            l_feats = list(sorted_theme_dict.values())
            l_classifier = torch.stack(l_feats)
            print(f"---> {level_name}'s classifier has a shape of {l_classifier.shape}")

            # Save the embeddings
            path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_{level_name}.npy")

            print(f'Saving to {path_save}')
            np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())
    elif args.dataset_name in ["coco", "oid", "lvis"] and args.aggregator != 'plain':

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)


        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_hrchy_l1.json"))

        theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

        theme_tree_features = defaultdict(dict)

        theme_feat_dict = defaultdict()
        for cat_id, entry in sorted(gpt_results.items(), key=lambda item: int(item[0])):
            if args.dataset_name in ["coco", "lvis"]:
                novel_list = coco_novel if args.dataset_name == "coco" else lvis_novel
                node_sentences = entry["candidate_sentences"] if entry['node_name'] in novel_list else [f"a {entry['node_name']}"]
            else:
                node_sentences = entry["candidate_sentences"]

            print(f"{entry['node_name']}: {len(node_sentences)}")

            node_tokens = clip.tokenize(node_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)
            node_features = F.normalize(node_features)
            # if node_features.size(0) == 1:
            #     print(f"level={level_name} cat={cat_id}")
            #     print(node_sentences)
            #     sys.exit()
            node_theme = theme_maker.get_theme(node_features)
            theme_feat_dict[cat_id] = node_theme

        total_num = len(list(theme_feat_dict.values()))
        print(f"Total feats = {total_num}")

        # Prepare and Save Features
        sorted_theme_dict = OrderedDict(sorted(theme_feat_dict.items(), key=lambda x: int(x[0])))

        l_feats = list(sorted_theme_dict.values())
        l_classifier = torch.stack(l_feats)
        print(f"---> {args.dataset_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_l1.npy")

        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())
    elif args.dataset_name in ["coco", "oid", "lvis"] and args.aggregator == 'plain':

        print('Loading CLIP')
        global_encoder, global_preprocess = clip.load(args.clip_model, device=device)

        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_hrchy_l1.json"))

        class_names = [entry["node_name"] for _, entry in sorted(gpt_results.items(), key=lambda item: int(item[0]))]
        acname_prompts = [f'a {cname}' for cname in class_names]


        classifier_tokens = clip.tokenize(acname_prompts).to(device)

        with torch.no_grad():
            cls_features = global_encoder.encode_text(classifier_tokens)

        l_classifier = F.normalize(cls_features)

        print(f"---> {args.dataset_name}'s classifier has a shape of {l_classifier.shape}")

        # Save the embeddings
        path_save = os.path.join(args.out_path, f"{args.dataset_name}_clip_hrchy_l1.npy")

        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())




