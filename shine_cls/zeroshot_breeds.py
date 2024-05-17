import argparse
import torch
import clip
from torch.nn import functional as F
from tqdm import tqdm
import torchmetrics
from data_utils.load_data import load_imagenet_val
from data_utils.cnames_imagenet import IMAGENET_CLASSES
from utils.themer import Themer
import time
from utils.fileios import *


def classification_acc(device, num_classes, top_k: int = 1):
  acc = torchmetrics.Accuracy(task="multiclass",
                              num_classes=num_classes,
                              top_k=top_k).to(device)
  return acc


def build_clip(model_size: str, device: str, jit: bool):
  # load model
  encoder, preprocesser = clip.load(model_size, device=device, jit=jit)
  encoder.eval()
  encoder.requires_grad_(False)
  return encoder, preprocesser


def do_clip_zeroshot(args, model, dloader, class_tree, label_mapper, templates=['a {}']):
  print("=== CLIP Zero-shot ===")
  class_tree.update(sorted(class_tree.items(), key=lambda item: int(item[0])))

  print("---> Generating classifier")
  class_names = [v["node_name"] for _, v in class_tree.items()]

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
  if txt_classifier.size(0) < 5:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
  else:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=5)

  total_time = 0.0
  num_images = 0

  for batch_idx, (og_images, og_labels) in enumerate(tqdm(dloader)):
    og_images = og_images.to(args.device)
    og_labels = og_labels.to(args.device)

    # ADAPTION TO BREEDS HIERARCHY
    # Create a mask for labels that exist in the label_mapper
    mask = torch.tensor([str(label.item()) in label_mapper for label in og_labels])
    # If none of the labels are in the mapper, skip this iteration
    if not mask.any():
      continue
    # Use the mask to get the filtered original images
    images = og_images[mask]
    # Use the mask to get the filtered original labels
    filtered_og_labels = og_labels[mask]
    # Map the original labels to the new ones for the filtered tensor
    labels = torch.tensor([label_mapper[str(label.item())] for label in filtered_og_labels])

    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    end_time = time.time()
    total_time += (end_time - start_time)
    num_images += labels.size(0)

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def do_shine(args, aggr_method, model, dloader, class_tree, label_mapper):
  print(f"=== SHiNe {aggr_method} ===")
  theme_maker = Themer(method=aggr_method, thresh=1, alpha=0.5)
  class_tree.update(sorted(class_tree.items(), key=lambda item: int(item[0])))

  print(f"\tGenerating classifier")
  zeroshot_weights = []
  for cat_id, entry in class_tree.items():
    texts = entry["candidate_sentences"]
    tokens = clip.tokenize(texts).to(args.device)
    txt_embeddings = model.encode_text(tokens)
    txt_embeddings = theme_maker.get_theme(txt_embeddings)
    zeroshot_weights.append(txt_embeddings)

  txt_classifier = torch.stack(zeroshot_weights)
  txt_classifier = F.normalize(txt_classifier)

  print(f"\tclassifier_dim = {txt_classifier.shape}")


  print("---> Evaluating")
  if txt_classifier.size(0) < 5:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
  else:
    acc_top1 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=1)
    acc_top5 = classification_acc(args.device, num_classes=int(txt_classifier.size(0)), top_k=5)

  total_time = 0.0
  num_images = 0

  for batch_idx, (og_images, og_labels) in enumerate(tqdm(dloader)):
    og_images = og_images.to(args.device)
    og_labels = og_labels.to(args.device)

    # ADAPTION TO BREEDS HIERARCHY
    # Create a mask for labels that exist in the label_mapper
    mask = torch.tensor([str(label.item()) in label_mapper for label in og_labels])
    # If none of the labels are in the mapper, skip this iteration
    if not mask.any():
      continue
    # Use the mask to get the filtered original images
    images = og_images[mask]
    # Use the mask to get the filtered original labels
    filtered_og_labels = og_labels[mask]
    # Map the original labels to the new ones for the filtered tensor
    labels = torch.tensor([label_mapper[str(label.item())] for label in filtered_og_labels])

    start_time = time.time()

    images = images.to(args.device)
    labels = labels.to(args.device)

    img_embeddings = model.encode_image(images)
    img_embeddings = F.normalize(img_embeddings)

    scores_clip = img_embeddings @ txt_classifier.T
    preds_clip = scores_clip.argmax(dim=1)

    acc_clip_ = acc_top1(scores_clip, labels)
    acc_top5_clip_ = acc_top5(scores_clip, labels)

    end_time = time.time()
    total_time += (end_time - start_time)
    num_images += labels.size(0)

  sec_per_item = total_time / num_images
  fps = 1.0 / sec_per_item

  return acc_top1.compute().item(), acc_top5.compute().item(), sec_per_item, fps


def print_results(method, results):
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
  # output_results = f"top1\ttop5\tSPI\tFPS\n{100 * results[0]},\t{100 * results[1]},\t{results[2]},\t{results[3]}"
  # output_path = method.replace(" ", "_").replace(".", "_").replace("/", "-")
  # dump_txt(f"output_breeds/{output_path}", output_results)
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
                      default="zeroshot",
                      choices=['zeroshot', 'shine']
                      )
  parser.add_argument('--breed_level',
                      type=str,
                      default='l1',
                      choices=['l1', 'l2', 'l3', 'l4', 'l5', 'l6']
                      )
  parser.add_argument('--hierarchy_root',
                      type=str,
                      default="hrchy_breeds",
                      )
  parser.add_argument('--batch_size',
                      type=int,
                      default=64,
                      )


  args = parser.parse_args()
  args.device = 'cuda'

  encoder, preprocesser = build_clip(args.model_size, args.device, jit=False)
  dset, dloader = load_imagenet_val(dataset_root=args.dataset_root, batch_size=args.batch_size, num_workers=8,
                                    shuffle=True)

  hier_paths = {
    'l1': f"{args.hierarchy_root}/composed_breed_l2_num_class=10.json",
    'l2': f"{args.hierarchy_root}/composed_breed_l3_num_class=29.json",
    'l3': f"{args.hierarchy_root}/composed_breed_l4_num_class=128.json",
    'l4': f"{args.hierarchy_root}/composed_breed_l5_num_class=466.json",
    'l5': f"{args.hierarchy_root}/composed_breed_l6_num_class=591.json",
    'l6': f"{args.hierarchy_root}/composed_breed_l7_num_class=98.json",
  }

  hier_mapper_paths = {
    'l1': f"{args.hierarchy_root}/mapper_l2_leaf2current.json",
    'l2': f"{args.hierarchy_root}/mapper_l3_leaf2current.json",
    'l3': f"{args.hierarchy_root}/mapper_l4_leaf2current.json",
    'l4': f"{args.hierarchy_root}/mapper_l5_leaf2current.json",
    'l5': f"{args.hierarchy_root}/mapper_l6_leaf2current.json",
    'l6': f"{args.hierarchy_root}/mapper_l7_leaf2current.json",
  }

  class_tree = load_json(hier_paths[args.breed_level])
  label_mapper = load_json(hier_mapper_paths[args.breed_level])

  # Baseline
  if args.method == "zeroshot":
    zeroshot_results = do_clip_zeroshot(args, encoder, dloader, class_tree, label_mapper, templates=['a {}'])
    print_results(
      method=f"CLIP-Zeroshot_baseline_{args.model_size}_BREEDS_{args.breed_level}", results=zeroshot_results,
    )
  # SHiNe
  elif args.method == "shine":
    shine_mean_results = do_shine(args, "mean", encoder, dloader, class_tree, label_mapper)

    print_results(
      method=f"SHiNe_Mean_{args.model_size}_BREEDS_{args.breed_level}", results=shine_mean_results
    )
  else:
    raise NotImplementedError(f"Method - {args.method} - is not supported!")


