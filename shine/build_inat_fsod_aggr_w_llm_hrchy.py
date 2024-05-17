import argparse
import torch
import numpy as np
import clip
from collections import defaultdict, OrderedDict
from shine.tools.themer import Themer
from shine.tools.fileios import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', default='inat', choices=['inat', 'fsod'])
    parser.add_argument('--gpt_results_root', default='inat_llm_answers')
    parser.add_argument('--prompter', default='isa', choices=['a', 'avg', 'concat', 'isa'])
    parser.add_argument('--aggregator', default='mean', choices=['peigen', 'mean', 'mixed', 'all_eigens'])
    parser.add_argument('--peigen_thresh', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--out_path', default='')
    parser.add_argument('--clip_model', default="RN50", choices=['ViT-B/32', 'RN50'])

    args = parser.parse_args()

    if not is_valid_folder(args.out_path): raise FileExistsError

    # Device Selection
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dataset_name == 'inat':
        level_names = ['l6', 'l5', 'l4', 'l3', 'l2', 'l1']
    else:
        level_names = ['l3', 'l2', 'l1']

    print('Loading CLIP')
    global_encoder, global_preprocess = clip.load(args.clip_model, device=device)
    theme_maker = Themer(method=args.aggregator, thresh=args.peigen_thresh, alpha=args.alpha)

    theme_tree_features = defaultdict(dict)
    for level_name in level_names:
        gpt_results = load_json(os.path.join(args.gpt_results_root,
                                             f"cleaned_{args.dataset_name}_gpt_hrchy_{level_name}.json"))
        for cat_id, entry in gpt_results.items():
            node_sentences = entry["candidate_sentences"]
            node_tokens = clip.tokenize(node_sentences).to(device)
            with torch.no_grad():
                node_features = global_encoder.encode_text(node_tokens)
            # node_features = F.normalize(node_features)
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
        # print(f'Saving to {path_save}')
        # torch.save(l_classifier.cpu(), path_save)
        print(f'Saving to {path_save}')
        np.save(open(path_save, 'wb'), l_classifier.cpu().numpy())




