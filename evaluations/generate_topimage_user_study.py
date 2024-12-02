import torch
import os
import json
import argparse

import pandas as pd
from dataset import get_concept_dataloader
from data import utils as data_utils
from model.cbm import Backbone, ConceptLayer, FinalLayer, NormalizationLayer
from torch import nn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_dir", type=str)
    parser.add_argument("output_filepath", type=str)
    parser.add_argument("--anno_dir", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=10)
    args = parser.parse_args()
    load_dir = args.load_dir
    output_filepath = args.output_filepath
    anno_dir = args.anno_dir
    k = args.top_k
    device = 'cuda'
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    if anno_dir is not None: args['annotation_dir'] = anno_dir 
    concept_file = os.path.join(load_dir, "concepts.txt")
    with open(concept_file) as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args["dataset"])


    # load models
    backbone = Backbone(args["backbone"], args["feature_layer"], device=device)
    concept_layer = ConceptLayer.from_pretrained(load_dir)
    final_layer = FinalLayer.from_pretrained(load_dir)
    normalize_layer = NormalizationLayer.from_pretrained(load_dir)

    model = nn.Sequential(backbone, concept_layer)
    model.eval()
    # get activations
    dataloader = get_concept_dataloader(args["dataset"], "val", concepts=concepts, preprocess=backbone.preprocess, shuffle=False)
    dataset = dataloader.dataset
    pil_data = data_utils.get_data(args["dataset"] + "_val")
    val_d_probe = args["dataset"] + "_val"
    cls_file = data_utils.LABEL_FILES[args["dataset"]]

    val_data_t = data_utils.get_data(val_d_probe, preprocess=backbone.preprocess)
    val_pil_data = data_utils.get_data(val_d_probe)
    val_loader = torch.utils.data.DataLoader(val_data_t, batch_size=128, shuffle=False, num_workers=8)
    # Collect concept activation values
    concepts_value = []
    model.to(device)
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            concept = model(x)
            concepts_value.append(concept)
    concepts_value = torch.cat(concepts_value)
    _, top_activated_images = concepts_value.topk(k=k, dim=0)
    data = {f"image_{i+1}_id":top_activated_images[i, :].cpu().numpy() for i in range(k)}
    data["description"] = concepts
    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_filepath, index=False)
