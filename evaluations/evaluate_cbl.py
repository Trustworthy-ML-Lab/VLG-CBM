import argparse
import os
import json

from data import utils as data_utils
from train_cbm import get_concept_dataloader, get_final_layer_dataset
from model.cbm import Backbone, BackboneCLIP, ConceptLayer, NormalizationLayer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    args = parser.parse_args()
    load_dir = args.path
     # Load arguments
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    with open(os.path.join(load_dir ,"concepts.txt"), 'r') as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)

    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(args.backbone, device=args.device, use_penultimate=args.use_clip_penultimate)
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    cbl = ConceptLayer.from_pretrained(load_dir, args.device)
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    train_concept_loader, test_concept_loader = get_final_layer_dataset(
        backbone, cbl, None, val_cbl_loader, save_dir=load_dir, load_dir=load_dir,
        batch_size=args.cbl_batch_size
    )
    normalization_layer = NormalizationLayer.from_pretrained(load_dir, device=args.device)