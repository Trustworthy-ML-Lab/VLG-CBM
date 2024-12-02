import argparse
import json
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import model.utils as utils
import data.utils as data_utils
from data.concept_dataset import get_concept_dataloader, get_final_layer_dataset
from glm_saga.elasticnet import IndexedTensorDataset, glm_saga
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    NormalizationLayer,
    load_cbm,
)

import numpy as np

MAX_GLM_STEP = 150
GLM_STEP_SIZE = 2 ** 0.1


def measure_acc(
    num_concepts,
    num_classes,
    num_samples,
    train_loader,
    val_loader,
    test_concept_loader,
    saga_step_size=0.1,
    saga_n_iters=500,
    device="cuda",
    max_lam=0.01,
    measure_level=(5, 10, 15, 20, 25, 30),
):
    linear = torch.nn.Linear(num_concepts, num_classes).to(device)
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = max_lam
    # Solve the GLM path
    max_sparsity = measure_level[-1] / num_concepts
    output_proj = glm_saga(linear, train_loader, saga_step_size, saga_n_iters, ALPHA, k=MAX_GLM_STEP, epsilon=1 / (GLM_STEP_SIZE ** MAX_GLM_STEP),
                    val_loader=val_loader, test_loader=test_concept_loader, do_zero=False, metadata=metadata, n_ex=num_samples, n_classes=num_classes,
                    max_sparsity=max_sparsity)
    path = output_proj['path']
    sparsity_list = [(params['weight'].abs() > 1e-5).float().mean().item() for params in path]

    # Measure accuracy on test set
    final_layer = torch.nn.Linear(num_concepts, num_classes)
    accs = []
    weights = []
    for eff_concept_num in measure_level:
        target_sparsity = eff_concept_num / num_concepts
        # Pick the lam with sparsity closest to target
        for i, sparsity in enumerate(sparsity_list):
            if sparsity >= target_sparsity:
                break
        params = path[i]
        W_g, b_g, lam = params["weight"], params["bias"], params["lam"]
        print(eff_concept_num, lam, sparsity)
        print(
            f"Num of effective concept: {eff_concept_num}. Choose lambda={lam:.6f} with sparsity {sparsity:.4f}"
        )
        W_g_trunc = utils.weight_truncation(W_g, target_sparsity)
        weight_contribs = torch.sum(torch.abs(W_g_trunc), dim=0)
        print(
            "Num concepts with outgoing weights:{}/{}".format(
                torch.sum(weight_contribs > 1e-5), len(weight_contribs)
            )
        )
        print(target_sparsity, (W_g_trunc.abs() > 0).sum())
        final_layer.load_state_dict({"weight": W_g_trunc, "bias": b_g})
        final_layer = final_layer.to(device)
        weights.append((W_g_trunc, b_g))
        # Test final weights
        correct = []
        for x, y in test_concept_loader:
            x, y = x.to(device), y.to(device)
            pred = final_layer(x).argmax(dim=-1)
            correct.append(pred == y)
        correct = torch.cat(correct)
        accs.append(correct.float().mean().item())
        print(f"Test Acc: {correct.float().mean():.4f}")
    print(f"Average acc: {sum(accs) / len(accs):.4f}")
    return path, {NEC: weight for NEC, weight in zip(measure_level, weights)}, accs


def sparsity_acc_test(load_dir, lam_max=0.1, bot_filter=0, anno=None):
    # Load arguments
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    if anno is not None:
        args.annotation_dir = anno
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)
    if anno is None:
        anno = args.annotation_dir
    # Concept filtering
    filtered_idx = None
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, device=args.device, use_penultimate=args.use_clip_penultimate
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)
    if os.path.exists(os.path.join(load_dir, "backbone.pt")):
        ckpt = torch.load(os.path.join(load_dir, "backbone.pt"))
        backbone.backbone.load_state_dict(ckpt)
    cbl = ConceptLayer.from_pretrained(load_dir, args.device)
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        val_split=args.val_split,
        seed=args.seed,
        label_dir=anno,
    )
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        val_split=args.val_split,
        seed=args.seed,
        label_dir=anno,
    )
    test_cbl_loader = get_concept_dataloader(
        args.dataset,
        "test",
        concepts,
        backbone.preprocess,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        val_split=None,
        seed=args.seed,
        label_dir=anno,
    )
    # Calculating test features
    train_concept_loader, val_concept_loader, _ = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        save_dir=load_dir,
        load_dir=load_dir
        if os.path.exists(os.path.join(load_dir, "train_concept_features.pt"))
        else None,
        batch_size=args.saga_batch_size,
        filter=filtered_idx,
    )
    normalization = NormalizationLayer.from_pretrained(load_dir, args.device)
    with torch.no_grad():
        test_concept_features = []
        test_concept_labels = []
        for features, _, labels in tqdm(test_cbl_loader):
            features = features.to(args.device)
            concept_logits = normalization(cbl(backbone(features)))
            test_concept_features.append(concept_logits.detach().cpu())
            test_concept_labels.append(labels)
        test_concept_features = torch.cat(test_concept_features, dim=0)
        concept_labels = torch.cat(test_concept_labels, dim=0)
    test_concept_dataset = TensorDataset(test_concept_features, concept_labels)
    test_concept_loader = DataLoader(
        test_concept_dataset, batch_size=args.saga_batch_size, shuffle=False
    )

    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        val_concept_loader,
        test_concept_loader,
        saga_step_size=args.saga_step_size,
        saga_n_iters=args.saga_n_iters,
        device=args.device,
        max_lam=lam_max,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"]["acc_test"] for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    # Save truncated weights
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs


def sparsity_acc_test_lf_cbm(load_dir, lam_max=0.1):
    # Load arguments
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)
        args = argparse.Namespace(**args)
    with open(os.path.join(load_dir, "concepts.txt"), "r") as f:
        concepts = f.read().split("\n")
    classes = data_utils.get_classes(args.dataset)
    # Concept filtering

    cbm = load_cbm(load_dir, args.device)
    cbm.eval()
    train_dataset = data_utils.get_data(args.dataset + "_train", preprocess=cbm.preprocess)
    test_dataset = data_utils.get_data(args.dataset + "_val", preprocess=cbm.preprocess)
    # Calculating test features
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, num_workers=8, batch_size=args.batch_size
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, shuffle=False, num_workers=8, batch_size=args.batch_size
    )
    with torch.no_grad():
        final_loaders = []
        for i, loader in enumerate([train_dataloader, test_dataloader]):
            concept_features = []
            concept_labels = []
            correct = 0
            for features, labels in tqdm(loader):
                features = features.to(args.device)
                pred, concept_logits = cbm(features)
                concept_features.append(concept_logits.detach().cpu())
                correct += (pred.argmax(dim=-1) == labels.to(args.device)).float().sum()
                concept_labels.append(labels)
            print("Accuracy: ", correct / len(loader.dataset))
            concept_features = torch.cat(concept_features, dim=0)
            concept_labels = torch.cat(concept_labels, dim=0)
            concept_dataset = (
                IndexedTensorDataset(concept_features, concept_labels)
                if i == 0
                else TensorDataset(concept_features, concept_labels)
            )
            concept_loader = DataLoader(
                concept_dataset, batch_size=args.saga_batch_size, shuffle=False
            )
            final_loaders.append(concept_loader)
    train_concept_loader, test_concept_loader = final_loaders
    path, truncated_weights, accs = measure_acc(
        len(concepts),
        len(classes),
        len(train_concept_loader.dataset),
        train_concept_loader,
        None,
        test_concept_loader,
        saga_n_iters=args.n_iters,
        device=args.device,
        max_lam=lam_max,
    )
    sparsity_list = [
        (params["weight"].abs() > 1e-5).float().mean().item() for params in path
    ]
    NEC = [len(concepts) * sparsity for sparsity in sparsity_list]
    acc = [params["metrics"]["acc_test"] for params in path]
    df = pd.DataFrame(data={"NEC": NEC, "Accuracy": acc})
    df.to_csv(os.path.join(load_dir, "metrics.csv"))
    # Save truncated weights
    for NEC in truncated_weights:
        W, b = truncated_weights[NEC]
        torch.save(W, os.path.join(load_dir, f"W_g@NEC={NEC:d}.pt"))
        torch.save(b, os.path.join(load_dir, f"b_g@NEC={NEC:d}.pt"))
    return accs
