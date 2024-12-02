import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.utils
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

import clip
from data import utils as data_utils
from glm_saga.elasticnet import glm_saga


class CBM_model(torch.nn.Module):
    def __init__(
        self, backbone_name, W_c, W_g, b_g, proj_mean, proj_std, device="cuda"
    ):
        super().__init__()
        model, preprocess = data_utils.get_target_model(backbone_name, device)
        self.preprocess = preprocess
        # remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        self.proj_layer = torch.nn.Linear(
            in_features=W_c.shape[1], out_features=W_c.shape[0], bias=False
        ).to(device)
        self.proj_layer.load_state_dict({"weight": W_c})

        self.proj_mean = proj_mean
        self.proj_std = proj_std

        self.final = torch.nn.Linear(
            in_features=W_g.shape[1], out_features=W_g.shape[0]
        ).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.proj_layer(x)
        proj_c = (x - self.proj_mean) / self.proj_std
        x = self.final(proj_c)
        return x, proj_c


class standard_model(torch.nn.Module):
    def __init__(self, backbone_name, W_g, b_g, proj_mean, proj_std, device="cuda"):
        super().__init__()
        model, _ = data_utils.get_target_model(backbone_name, device)
        # remove final fully connected layer
        if "clip" in backbone_name:
            self.backbone = model
        elif "cub" in backbone_name:
            self.backbone = lambda x: model.features(x)
        else:
            self.backbone = torch.nn.Sequential(*list(model.children())[:-1])

        self.proj_mean = proj_mean
        self.proj_std = proj_std

        self.final = torch.nn.Linear(
            in_features=W_g.shape[1], out_features=W_g.shape[0]
        ).to(device)
        self.final.load_state_dict({"weight": W_g, "bias": b_g})
        self.concepts = None

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        proj_c = (x - self.proj_mean) / self.proj_std
        x = self.final(proj_c)
        return x, proj_c


class Backbone(nn.Module):
    # store intermediate feature values from backbone
    feature_vals = {}

    def __init__(self, backbone_name: str, feature_layer: str, device: str = "cuda"):
        super().__init__()
        target_model, target_preprocess = data_utils.get_target_model(
            backbone_name, device
        )

        # hook into feature layer
        def hook(module, input, output):
            self.feature_vals[output.device] = output

        command = "target_model.{}.register_forward_hook(hook)".format(feature_layer)
        eval(command)

        # assign backbone and preprocess
        self.backbone = target_model
        self.preprocess = target_preprocess
        self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[backbone_name]

    def forward(self, x):
        out = self.backbone(x)
        return self.feature_vals[out.device].mean(dim=[2, 3])

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["feature_layer"], device)


class BackboneCLIP(nn.Module):
    def __init__(
        self, backbone_name: str, use_penultimate: bool = True, device: str = "cuda"
    ):
        super().__init__()
        target_model, target_preprocess = clip.load(backbone_name[5:], device=device)
        if use_penultimate:
            logger.info("Using penultimate layer of CLIP")
            target_model = target_model.visual
            N = target_model.attnpool.c_proj.in_features
            identity = torch.nn.Linear(N, N, dtype=torch.float16, device=device)
            nn.init.zeros_(identity.bias)
            identity.weight.data.copy_(torch.eye(N))
            target_model.attnpool.c_proj = identity
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{backbone_name}_penultimate"
            ]
        else:
            logger.info("Using final layer of CLIP")
            target_model = target_model.visual
            self.output_dim = data_utils.BACKBONE_ENCODING_DIMENSION[backbone_name]

        # assign backbone and preprocess
        self.backbone = target_model.float()
        self.preprocess = target_preprocess

    def forward(self, x):
        output = self.backbone(x).float()
        return output

    def save_model(self, save_dir):
        torch.save(self.backbone.state_dict(), os.path.join(save_dir, "backbone.pt"))

    @classmethod
    def from_args(cls, load_dir: str, device: str = "cuda"):
        with open(os.path.join(load_dir, "args.txt"), "r") as f:
            args = json.load(f)
        return cls(args["backbone"], args["use_clip_penultimate"], device)

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        model = cls.from_args(load_path, device)
        model.backbone.load_state_dict(
            torch.load(os.path.join(load_path, "backbone.pt"))
        )
        return model


class ConceptLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_hidden: int = 0,
        bias: bool = True,
        device: str = "cuda",
    ):
        super().__init__()
        model = [nn.Linear(in_features, out_features, bias=bias)]
        for _ in range(num_hidden):
            model.append(nn.ReLU())
            model.append(nn.Linear(out_features, out_features, bias=bias))

        self.model = nn.Sequential(*model).to(device)
        self.out_features = out_features
        logger.info(self.model)

    def forward(self, x):
        return self.model(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "cbl.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_hidden = args["cbl_hidden_layers"]
        if args["use_clip_penultimate"] and args["backbone"].startswith("clip"):
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[
                f"{args['backbone']}_penultimate"
            ]
        else:
            encoder_dim = data_utils.BACKBONE_ENCODING_DIMENSION[args["backbone"]]
        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))

        # create model
        model = cls(encoder_dim, num_concepts, num_hidden=num_hidden, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "cbl.pt")))
        return model


class NormalizationLayer(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, device: str = "cuda"):
        super().__init__()
        self.mean = mean.to(device)
        self.std = std.to(device)

    def forward(self, x):
        return (x - self.mean) / self.std

    def save_model(self, save_dir):
        # save model
        torch.save(self.mean, os.path.join(save_dir, "train_concept_features_mean.pt"))
        torch.save(self.std, os.path.join(save_dir, "train_concept_features_std.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        mean = torch.load(
            os.path.join(load_path, "train_concept_features_mean.pt"),
            map_location=device,
        )
        std = torch.load(
            os.path.join(load_path, "train_concept_features_std.pt"),
            map_location=device,
        )
        normalization_layer = cls(mean, std, device=device)
        return normalization_layer


class FinalLayer(nn.Linear):
    def __init__(self, in_features: int, out_features: int, device: str = "cuda"):
        super().__init__(in_features, out_features, bias=True)
        self.to(device)

    def forward(self, x):
        return super().forward(x)

    def save_model(self, save_dir):
        # save model
        torch.save(self.state_dict(), os.path.join(save_dir, "final.pt"))

    @classmethod
    def from_pretrained(cls, load_path: str, device: str = "cuda"):
        # load args
        with open(os.path.join(load_path, "args.txt"), "r") as f:
            args = json.load(f)

        num_concepts = len(data_utils.get_concepts(f"{load_path}/concepts.txt"))
        num_classes = len(data_utils.get_classes(args["dataset"]))

        # create model
        model = cls(num_concepts, num_classes, device=device)
        model.load_state_dict(torch.load(os.path.join(load_path, "final.pt")))
        return model


def load_cbm(load_dir, device):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)

    W_c = torch.load(os.path.join(load_dir, "W_c.pt"), map_location=device)
    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = CBM_model(args["backbone"], W_c, W_g, b_g, proj_mean, proj_std, device)
    return model


def load_std(load_dir, device):
    with open(os.path.join(load_dir, "args.txt"), "r") as f:
        args = json.load(f)

    W_g = torch.load(os.path.join(load_dir, "W_g.pt"), map_location=device)
    b_g = torch.load(os.path.join(load_dir, "b_g.pt"), map_location=device)

    proj_mean = torch.load(os.path.join(load_dir, "proj_mean.pt"), map_location=device)
    proj_std = torch.load(os.path.join(load_dir, "proj_std.pt"), map_location=device)

    model = standard_model(args["backbone"], W_g, b_g, proj_mean, proj_std, device)
    return model


def per_class_accuracy(
    model: torch.nn.Module, loader: DataLoader, classes: List[str], device: str = "cuda"
) -> Dict[str, float]:
    correct = torch.zeros(len(classes)).to(device)
    total = torch.zeros(len(classes)).to(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for features, _, targets in tqdm(loader):
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            preds = logits.argmax(dim=1)
            for pred, target in zip(preds, targets):
                total[target] += 1
                if pred == target:
                    correct[target] += 1

    per_class_accuracy = correct / total
    total_accuracy = correct.sum() / total.sum()
    total_datapoints = total.sum()

    # return a dictionary of class names and accuracies, and total accuracy
    return {
        "Per class accuracy": {
            classes[i]: f"{per_class_accuracy[i]*100.0:.2f}"
            for i in range(len(classes))
        },
        "Overall accuracy": f"{total_accuracy*100.0:.2f}",
        "Datapoints": f"{total_datapoints}",
    }


def validate_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    val_loader: DataLoader,
    loss_fn: torch.nn.Module,
    device: str = "cuda",
):
    val_loss = 0.0
    with torch.no_grad():
        logger.info("Running CBL validation")
        for features, concept_one_hot, _ in tqdm(val_loader):
            features = features.to(device)
            concept_one_hot = concept_one_hot.to(device)

            # forward pass
            concept_logits = cbl(backbone(features))

            # calculate loss
            batch_loss = loss_fn(concept_logits, concept_one_hot)
            val_loss += batch_loss.item()

        # finalize metrics and update model
        val_loss = val_loss / len(val_loader)

    return val_loss


def train_cbl(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    loss_fn: torch.nn.Module,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    concepts: Optional[List[str]] = None,
    tb_writer=None,
    device: str = "cuda",
    finetune: bool = False,
    optimizer: str = "sgd",
    scheduler: str = None,
    backbone_lr: float = 1e-3,
    data_parallel=False,
):
    # setup optimizer
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(
            cbl.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9
        )
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(cbl.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError
    if finetune:
        optimizer.add_param_group({"params": backbone.parameters(), "lr": backbone_lr})

    # setup schedular
    if scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_val_loss = float("inf")
    best_val_loss_epoch = None
    best_model_state = None
    if data_parallel:
        backbone = torch.nn.DataParallel(backbone)
        cbl = torch.nn.DataParallel(cbl)
    for epoch in range(epochs):
        train_loss = 0
        lr = optimizer.param_groups[0]["lr"]

        logger.info(f"Running CBL training for Epoch: {epoch}")
        its = tqdm(total=len(train_loader), position=0, leave=True)
        for batch_idx, (features, concept_one_hot, _) in enumerate(train_loader):
            features = features.to(device)  # (batch_size, feature_dim)
            concept_one_hot = concept_one_hot.to(device)  # (batch_size, n_concepts)

            # forward pass
            if finetune:
                backbone.train()
                embeddings = backbone(features)
            else:
                with torch.no_grad():
                    embeddings = backbone(features)  # (batch_size, feature_dim)
            concept_logits = cbl(embeddings)  # (batch_size, n_concepts)

            # calculate loss
            batch_loss = loss_fn(concept_logits, concept_one_hot)
            train_loss += batch_loss.item()

            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            # print batch stats
            if (batch_idx + 1) % 1000 == 0:
                its.update(1000)
                print(
                    "Epoch: {} | Batch: {} | Loss: {:.6f}".format(
                        epoch, batch_idx, batch_loss.item()
                    )
                )

                # exit if loss is nan
                if torch.isnan(batch_loss):
                    # Exit process if loss is nan
                    logger.error(f"Loss is nan at epoch {epoch} and batch {batch_idx}")
                    sys.exit(1)
        backbone.eval()
        # finalize metrics and update model
        its.close()
        train_loss = train_loss / len(train_loader)
        # train_per_concept_roc = train_per_concept_roc.compute()

        # evaluate on validation set
        logger.info(f"Running CBL validation for Epoch: {epoch}")
        val_loss = validate_cbl(
            backbone,
            cbl.module if data_parallel else cbl,
            val_loader,
            loss_fn=loss_fn,
            device=device,
        )
        if val_loss < best_val_loss:
            logger.info(f"Updating best val loss at epoch: {epoch}")
            best_val_loss = val_loss
            best_val_loss_epoch = epoch
            best_backbone_state = backbone.state_dict()
            best_model_state = cbl.state_dict()

        # write to tensorboard
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("lr", lr, epoch)

        # print epoch stats
        logger.info(
            f"Epoch: {epoch} | Train loss: {train_loss:.6f} | Val loss: {val_loss:.6f}"
        )

        # Step the scheduler
        if scheduler is not None:
            scheduler.step(val_loss)

    # return best model based on validation loss
    logger.info(f"Best val loss: {best_val_loss:.6f} at epoch {best_val_loss_epoch}")
    cbl.load_state_dict(best_model_state)
    backbone.load_state_dict(best_backbone_state)
    if data_parallel:
        cbl = cbl.module
        backbone = backbone.module
    return cbl, backbone


def test_model(
    loader: DataLoader,
    backbone: Backbone,
    cbl: ConceptLayer,
    normalization: NormalizationLayer,
    final_layer: FinalLayer,
    device: str = "cuda",
):
    acc_mean = 0.0
    for features, concept_one_hot, targets in tqdm(loader):
        features = features.to(device)
        concept_one_hot = concept_one_hot.to(device)
        targets = targets.to(device)

        # forward pass
        with torch.no_grad():
            embeddings = backbone(features)
            concept_logits = cbl(embeddings)
            concept_probs = normalization(concept_logits)
            logits = final_layer(concept_probs)

        # calculate accuracy
        preds = logits.argmax(dim=1)
        accuracy = (preds == targets).sum().item()
        acc_mean += accuracy

    return acc_mean / len(loader.dataset)


def train_sparse_final(
    linear,
    indexed_train_loader,
    val_loader,
    n_iters,
    lam,
    step_size=0.1,
    device="cuda",
):
    # zero initialize
    num_classes = linear.weight.shape[0]
    linear.weight.data.zero_()
    linear.bias.data.zero_()

    ALPHA = 0.99
    metadata = {}
    metadata["max_reg"] = {}
    metadata["max_reg"]["nongrouped"] = lam

    # Solve the GLM path
    output_proj = glm_saga(
        linear,
        indexed_train_loader,
        step_size,
        n_iters,
        ALPHA,
        epsilon=1,
        k=1,
        val_loader=val_loader,
        do_zero=False,
        metadata=metadata,
        n_ex=len(indexed_train_loader.dataset),
        n_classes=num_classes,
        verbose=True,
    )

    return output_proj


def train_dense_final(
    model,
    indexed_train_loader,
    val_loader,
    n_iters,
    lr=0.001,
    device="cuda",
):
    # setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # setup schedular
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    # setup loss
    ce_loss = torch.nn.CrossEntropyLoss()

    # train
    for epoch in range(n_iters):
        train_loss = 0
        val_loss = 0
        val_accuracy = 0

        # train
        for inputs, targets, _ in tqdm(indexed_train_loader, desc="Train"):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # forward
            logits = model(inputs)

            # calculate loss
            batch_loss = ce_loss(logits, targets)
            train_loss += batch_loss.item()

            # optimize
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        train_loss = train_loss / len(indexed_train_loader)

        # validation
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # forward
                logits = model(inputs)

                # calculate loss
                batch_loss = ce_loss(logits, targets)
                val_loss += batch_loss.item()

                # calculate metrics
                classes = torch.argmax(logits, dim=1)
                val_accuracy += (classes == targets).sum().item()
        val_accuracy = val_accuracy / len(val_loader.dataset) * 100.0
        val_loss = val_loss / len(val_loader)

        # print stats
        print(
            f"Epoch: {epoch}, Train loss: {train_loss}, Val loss: {val_loss}, Val acc: {val_accuracy}, lr: {optimizer.param_groups[0]['lr']}"
        )

        # Step the scheduler
        scheduler.step()

    output_proj = {}
    output_proj["path"] = [{}]
    output_proj["path"][0]["weight"] = model.weight
    output_proj["path"][0]["bias"] = model.bias
    output_proj["path"][0]["lr"] = lr
    for key in ("lam", "alpha", "time"):
        output_proj["path"][0][key] = -1.0
    output_proj["path"][0]["metrics"] = {"val_accuracy": val_accuracy}
    return output_proj
