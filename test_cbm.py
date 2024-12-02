import argparse
import json
import os
import random
import numpy as np
import torch
from loguru import logger

from data import utils as data_utils
from model.cbm import Backbone, BackboneCLIP, ConceptLayer, FinalLayer, NormalizationLayer, test_model
from dataset import get_concept_dataloader

parser = argparse.ArgumentParser(description="Settings for creating CBM")
parser.add_argument("--load_dir", type=str, default=None, help="where to load trained models from")

def testcbm(args):
    logger.info("Loading CBL from {}".format(args.load_dir))
    
    # Load classes
    _ = data_utils.get_classes(args.dataset)

    # Load Backbone model
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device)
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    if os.path.exists(os.path.join(args.load_dir, "backbone.pt")):
        ckpt = torch.load(os.path.join(args.load_dir, "backbone.pt"))
        backbone.backbone.load_state_dict(ckpt)
    # load concepts set directly from load model
    with open(os.path.join(args.load_dir, "concepts.txt"), 'r') as f:
        concepts = f.read().split("\n")
    # get test loader
    test_cbl_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        preprocess=backbone.preprocess,
        val_split=None,  # not needed
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    
    # get model
    cbl = ConceptLayer.from_pretrained(args.load_dir, args.device)
    normalization_layer = NormalizationLayer.from_pretrained(args.load_dir, args.device)
    final_layer = FinalLayer.from_pretrained(args.load_dir, args.device)

    # get test accuracy
    test_accuracy = test_model(
        test_cbl_loader, backbone, cbl, normalization_layer, final_layer, args.device
    )
    logger.info(f"Test accuracy: {test_accuracy}")

if __name__ == "__main__":
    # run the training
    args = parser.parse_args()
    
    # load args file from load_dir
    load_dir = args.load_dir
    with open(os.path.join(args.load_dir, "args.txt")) as f:
        loaded_args = json.load(f)
    for key, value in loaded_args.items():
        setattr(args, key, value)
    setattr(args, "load_dir", load_dir)
    logger.info(args)

    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # test the model
    testcbm(args)
