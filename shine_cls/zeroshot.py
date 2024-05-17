import argparse
import torch
import clip
from torch.nn import functional as F
from tqdm import tqdm
import numpy as np
import torchmetrics
from data_utils.load_data import load_imagenet_val
from data_utils.cnames_imagenet import IMAGENET_CLASSES
from utils.themer import Themer
import json
import sys
import time
from utils.fileios import *

DEBUG_MODE = False
COUNT_RELATIVES = False

def classification_acc(device, top_k: int = 1):
  acc = torchmetrics.Accuracy(task="multiclass",
                              num_classes=1000,
                              top_k=top_k).to(device)
  return acc


def build_clip(model_size: str, device: str, jit: bool):
  # load model
  encoder, preprocesser = clip.load(model_size, device=device, jit=jit)
  encoder.eval()
  encoder.requires_grad_(False)
  return encoder, preprocesser


def do_clip_zeroshot(args, model, dloader, class_names, templates=['a {}']):
  print("=== CLIP Zero-shot ===")

  print("---> Generating classifier")
  zeroshot_weights = []
  for classname in tqdm(class_names):
    texts = [template.format(classname) for template in templates]
    tokens = clip.tokenize(texts).to(args.device)
    txt_embeddings = model.encode_text(tokens)
    txt_embeddings = torch.mean(txt_embeddings, dim=0)
    zeroshot_weights.append(txt_embeddings)

  txt_classifier = torch.stack(zeroshot_weights)
  txt_classifier = F.normalize(txt_classifier)

  print(f"\tclassifier_dim = {txt_classifier.shape}")


  print("---> Evaluating")
  acc_top1 = classification_acc(args.device, top_k=1)
  acc_top5 = classification_acc(args.device, top_k=5)

  total_time = 0.0
  num_images = 1000 if DEBUG_MODE else len(dloader)

  for batch_idx, (images, labels) in enumerate(tqdm(dloader)):
    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)
    end_time = time.time()

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    total_time += (end_time - start_time)
    if DEBUG_MODE and batch_idx >= num_images-1:
      break

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def do_shine(args, aggr_method, model, dloader, class_tree, template='a {}'):
  print(f"=== SHiNe {aggr_method} ===")
  theme_maker = Themer(method=aggr_method, thresh=1, alpha=0.5)

  print(f"\tGenerating classifier")
  zeroshot_weights = []
  for cat_id, entry in tqdm(sorted(class_tree.items(), key=lambda item: int(item[0]))):
    texts = entry["candidate_sentences"]
    texts = [template.format(t) for t in texts]

    tokens = clip.tokenize(texts).to(args.device)
    txt_embeddings = model.encode_text(tokens)
    txt_embeddings = theme_maker.get_theme(txt_embeddings)
    zeroshot_weights.append(txt_embeddings)

  txt_classifier = torch.stack(zeroshot_weights)
  txt_classifier = F.normalize(txt_classifier)

  print(f"\tclassifier_dim = {txt_classifier.shape}")

  print("---> Evaluating")
  acc_top1 = classification_acc(args.device, top_k=1)
  acc_top5 = classification_acc(args.device, top_k=5)
  total_time = 0.0
  num_images = 1000 if DEBUG_MODE else len(dloader)

  for batch_idx, (images, labels) in enumerate(tqdm(dloader)):
    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)
    end_time = time.time()

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    total_time += (end_time - start_time)
    if DEBUG_MODE and batch_idx >= num_images-1:
      break

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def print_results(method, results, hierarchy_source):
  method_name = method.upper()
  print("\n")
  print("=" * 25 + f" {method_name}-based Final Results " + "=" * 25)
  print("\n")
  print(f"[Classification]")
  print(f"Top-1 Acc   : {100 * results[0]}")
  print(f"Top-5 Acc   : {100 * results[1]}")
  print(f"[Speed]")
  print(f"Sec per Item: {results[2]} secs")
  print(f"FPS         : {results[3]} fps")
  print("=" * 25 + "          END          " + "=" * 25)
  output_results = f"top1\ttop5\tSPI\tFPS\n{100 * results[0]},\t{100 * results[1]},\t{results[2]},\t{results[3]}"
  # output_path = method.replace(" ", "_").replace(".", "_").replace("/", "-")
  # output_path = output_path + "_" + hierarchy_source.replace("hierarchy/", "").replace(".json", "")
  # dump_txt(f"output_speed/{output_path}", output_results)
  # print(f"Succ. dumped experiment results to: "+f"output_speed/{output_path}")


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='shine_classification',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--dataset_root',
                      type=str,
                      default="../datasets/imagenet2012/",
                      )
  parser.add_argument('--model_size',
                      type=str,
                      default='ViT-B/32',
                      choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'],
                      )
  parser.add_argument('--method',
                      type=str,
                      default="shine",
                      choices=['zeroshot', 'shine'],
                      )
  parser.add_argument('--hierarchy_tree_path',
                      type=str,
                      default="hrchy_imagenet1k/imagenet1k_hrchy_llm_composed.json",
                      choices=[
                        "hrchy_imagenet1k/imagenet1k_hrchy_wordnet.json",         # WordNet hierarchy
                        "hrchy_imagenet1k/imagenet1k_hrchy_llm_composed.json",    # LLM-generated hierarchy
                      ],
                      )
  parser.add_argument('--batch_size',
                      type=int,
                      default=1,
                      )
  parser.add_argument('--num_runs',
                      type=int,
                      default=1,
                      )

  args = parser.parse_args()
  args.device = 'cuda'

  encoder, preprocesser = build_clip(args.model_size, args.device, jit=False)
  dset, dloader = load_imagenet_val(dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=8,
                                    shuffle=False if args.num_runs > 1 else True    # do not shuffle for testing FPS
                                    )

  class_tree = load_json(args.hierarchy_tree_path)
  print(f"Loaded hierarchy tree from: {args.hierarchy_tree_path}")

  # Baseline
  if args.method == "zeroshot":
    num_runs = args.num_runs
    total_fps = 0

    for i in range(num_runs):
      zeroshot_results = do_clip_zeroshot(args, encoder, dloader, IMAGENET_CLASSES, templates=['a {}'])
      total_fps += zeroshot_results[-1]

    zeroshot_results = list(zeroshot_results)
    zeroshot_results[-1] = total_fps / num_runs
    print_results(method=f"CLIP-Zeroshot_w_{args.model_size}", results=zeroshot_results,
                  hierarchy_source=args.hierarchy_tree_path)
  elif args.method == "shine":
    num_runs = args.num_runs
    total_fps = 0

    for i in range(num_runs):
      shine_mean_results = do_shine(args, "mean", encoder, dloader, class_tree, template='a {}')
      total_fps += shine_mean_results[-1]

    shine_mean_results = list(shine_mean_results)
    shine_mean_results[-1] = total_fps / num_runs
    print_results(method=f"SHiNe-Mean_w_{args.model_size}", results=shine_mean_results,
                  hierarchy_source=args.hierarchy_tree_path)
  else:
    raise NotImplementedError(f"Method - {args.method} - is not supported!")


