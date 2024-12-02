import json
import os
import torch
import torch.nn as nn
import argparse
import pandas as pd
from tqdm import tqdm
from typing import List

import model.cbm as cbm
from data import utils as data_utils
import visualization.plots as plots
from model.cbm import Backbone, ConceptLayer, FinalLayer, NormalizationLayer
from dataset import get_concept_dataloader, ConceptDataset



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("load_dir", type=str)
    parser.add_argument("output_filepath", type=str)
    parser.add_argument("--NEC", type=int, default=None)
    parser.add_argument("--lf_cbm", action='store_true')
    args = parser.parse_args()
    NEC = args.NEC
    device = "cuda"
    load_dir = args.load_dir
    output_filepath = args.output_filepath
    use_lf_cbm = args.lf_cbm
    with open(os.path.join(load_dir ,"args.txt"), 'r') as f:
        args = json.load(f)
    concept_file = os.path.join(load_dir, "concepts.txt")
    with open(concept_file) as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args["dataset"])
    # load models
    if use_lf_cbm:
        cbm_model = cbm.load_cbm(load_dir, device=device)
        cbm_model.eval()
        preprocess = cbm_model.preprocess
        final_weights = cbm_model.final.weight.to(device)
        final_bias = cbm_model.final.bias.to(device)
        model = lambda x: cbm_model(x)[1]
    else:
        backbone = Backbone.from_pretrained(load_dir, device=device)
        concept_layer = ConceptLayer.from_pretrained(load_dir, device=device)
        final_layer = FinalLayer.from_pretrained(load_dir, device=device)
        final_weights = final_layer.weight.to(device)
        final_bias = final_layer.bias.to(device)
        normalize_layer = NormalizationLayer.from_pretrained(load_dir, device=device)
        preprocess = backbone.preprocess
        model = nn.Sequential(backbone, concept_layer, normalize_layer)
        model.eval()
    if NEC is not None:
        final_weights = torch.load(os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"), map_location=device)
    # get activations
    dataset = data_utils.get_data(args["dataset"] + "_val", preprocess=preprocess)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, batch_size=32, num_workers=8)

    gt_list, pred_list, attrib_list, contrib_list, remain_contrib_list = [], [], [], [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            c_prediction = model(x)
            # contributions: n_samples * n_concepts * n_classes
            contributions = c_prediction[:, :, None] * final_weights.T[None, :, :] 
            final_predictions = (contributions.sum(dim=1) + final_bias).argmax(dim=-1)
            decision_contributions = contributions[torch.arange(len(final_predictions)), :, final_predictions]
            top_contrib, top_concepts = decision_contributions.topk(k=5, dim=-1)
            remain_contrib = decision_contributions.sum(dim=1) - top_contrib.sum(dim=1)
            # Process concepts with negative activation: mark them as -i-1
            concepts_is_neg = c_prediction.gather(1, top_concepts) < 0
            top_concepts[concepts_is_neg] = -(top_concepts[concepts_is_neg] + 1)
            gt_list.append(y)
            pred_list.append(final_predictions)
            attrib_list.append(top_concepts)
            contrib_list.append(top_contrib)
            remain_contrib_list.append(remain_contrib)
    # Aggregate results and sanity check
    gt_all = torch.cat(gt_list)
    pred_all = torch.cat(pred_list)
    attrib_all = torch.cat(attrib_list)
    contrib_all = torch.cat(contrib_list)
    remain_all = torch.cat(remain_contrib_list)
    print(attrib_all.shape)
    print(f"Prediction accuracy: {(gt_all == pred_all).float().mean()}")
    # Save data
    data = {}
    data["ground_truth"] = [classes[i] for i in gt_all.cpu().numpy()]
    data["prediction"] = [classes[i] for i in pred_all.cpu().numpy()]
    data.update({f"concept_{i+1}": [(concepts[j] if j >= 0 else "NOT " + concepts[-j-1])for j in attrib_all[:, i]] for i in range(5)})
    data.update({f"contrib_{i+1}": contrib_all[:, i].cpu().numpy() for i in range(5)})
    data["image_idx"] = [i for i in range(len(gt_all))]
    data["remain_contrib"] = remain_all.cpu().numpy()
    df = pd.DataFrame.from_dict(data)
    df.to_csv(output_filepath)