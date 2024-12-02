import argparse
import datetime
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model.utils as utils
from data import utils as data_utils
from data.concept_dataset import (
    get_concept_dataloader,
    get_filtered_concepts_and_counts,
    get_final_layer_dataset,
)
from loss import get_loss
from model.cbm import (
    Backbone,
    BackboneCLIP,
    ConceptLayer,
    FinalLayer,
    per_class_accuracy,
    test_model,
    train_cbl,
    train_dense_final,
    train_sparse_final,
)


class LoggerWriter:
    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message.rstrip() != "":
            logger.log(self.level, message.rstrip())

    def flush(self):
        pass


def train_cbm_and_save(args):
    # Setup log directory and logger
    save_dir = "{}/{}_cbm_{}".format(
        args.save_dir,
        args.dataset,
        datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S"),
    )
    while os.path.exists(save_dir):
        save_dir += "-1"
    os.makedirs(save_dir)
    logger.add(
        os.path.join(save_dir, "train.log"),
        format="{time} {level} {message}",
        level="DEBUG",
    )
    logger.info("Saving model to {}".format(save_dir))
    with open(os.path.join(save_dir, "args.txt"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # Load classes
    classes = data_utils.get_classes(args.dataset)

    # Load Backbone model
    if args.backbone.startswith("clip_"):
        backbone = BackboneCLIP(
            args.backbone, use_penultimate=args.use_clip_penultimate, device=args.device
        )
    else:
        backbone = Backbone(args.backbone, args.feature_layer, args.device)

    # Remove concepts that are not present in the annotations
    if args.load_dir is None:
        if args.skip_concept_filter:
            logger.info("Skipping concept filtering")
            concepts, concept_counts = data_utils.load_concept_and_count(
                os.path.dirname(args.concept_set), filter_file=args.filter_set
            )
        else:
            # filter concepts
            logger.info("Filtering concepts")
            raw_concepts = data_utils.get_concepts(args.concept_set, args.filter_set)
            (
                concepts,
                concept_counts,
                filtered_concepts,
            ) = get_filtered_concepts_and_counts(
                args.dataset,
                raw_concepts,
                preprocess=backbone.preprocess,
                val_split=args.val_split,
                batch_size=args.cbl_batch_size,
                num_workers=args.num_workers,
                confidence_threshold=args.cbl_confidence_threshold,
                label_dir=args.annotation_dir,
                use_allones=args.allones_concept,
                seed=args.seed,
            )

            # save concept counts
            data_utils.save_concept_count(concepts, concept_counts, save_dir)
            data_utils.save_filtered_concepts(filtered_concepts, save_dir)
    else:
        # load concepts set directly from load model
        logger.info("Loading concepts from {}".format(args.load_dir))
        concepts, concept_counts = data_utils.load_concept_and_count(
            args.load_dir, filter_file=args.filter_set
        )

    with open(os.path.join(save_dir, "concepts.txt"), "w") as f:
        f.write(concepts[0])
        for concept in concepts[1:]:
            f.write("\n" + concept)

    # setup tensorboard writer
    tb_writer = SummaryWriter(save_dir)

    # setup all dataloaders
    augmented_train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=True,  # shuffle for training
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=args.crop_to_concept_prob,  # crop to concept
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
    train_cbl_loader = get_concept_dataloader(
        args.dataset,
        "train",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,  # no shuffle to match order
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )  # no shuffle to match labels
    val_cbl_loader = get_concept_dataloader(
        args.dataset,
        "val",
        concepts,
        preprocess=backbone.preprocess,
        val_split=args.val_split,
        batch_size=args.cbl_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        confidence_threshold=args.cbl_confidence_threshold,
        crop_to_concept_prob=0.0,  # no augmentation
        label_dir=args.annotation_dir,
        use_allones=args.allones_concept,
        seed=args.seed,
    )
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

    ##############################################
    # CBL training: Train CBL to map backbone features to concept space
    ##############################################
    loss_fn = get_loss(
        args.cbl_loss_type,
        len(concepts),
        len(train_cbl_loader.dataset),
        concept_counts,
        args.cbl_pos_weight,
        args.cbl_auto_weight,
        args.cbl_twoway_tp,
        args.device,
    )

    if args.load_dir is None:
        logger.info("Training CBL")
        cbl = ConceptLayer(
            backbone.output_dim,
            len(concepts),
            num_hidden=args.cbl_hidden_layers,
            device=args.device,
        )
        cbl, backbone = train_cbl(
            backbone,
            cbl,
            augmented_train_cbl_loader,
            val_cbl_loader,
            args.cbl_epochs,
            loss_fn=loss_fn,
            lr=args.cbl_lr,
            weight_decay=args.cbl_weight_decay,
            concepts=concepts,
            tb_writer=tb_writer,
            device=args.device,
            finetune=args.cbl_finetune,
            optimizer=args.cbl_optimizer,
            scheduler=args.cbl_scheduler,
            backbone_lr=args.cbl_lr * args.cbl_bb_lr_rate,
            data_parallel=args.data_parallel,
        )
    else:
        logger.info("Loading CBL from {}".format(args.load_dir))
        cbl = ConceptLayer.from_pretrained(args.load_dir, args.device)
        if args.backbone.startswith("clip_"):
            raise NotImplementedError(
                "Loading backbone from pretrained model is not supported yet"
            )
        else:
            backbone = Backbone.from_pretrained(args.load_dir, args.device)

    cbl.save_model(save_dir)
    if args.cbl_finetune:
        backbone.save_model(save_dir)

    ##############################################
    # FINAL layer training
    ##############################################
    (
        train_concept_loader,
        val_concept_loader,
        normalization_layer,
    ) = get_final_layer_dataset(
        backbone,
        cbl,
        train_cbl_loader,
        val_cbl_loader,
        save_dir,
        load_dir=args.load_dir,
        batch_size=args.saga_batch_size,
        device=args.device,
    )

    # Make linear model
    final_layer = FinalLayer(len(concepts), len(classes), device=args.device)

    if args.dense:
        logger.info(f"Training dense final layer with lr: {args.dense_lr} ...")
        output_proj = train_dense_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.dense_lr,
            device=args.device,
        )
    else:
        logger.info(f"Training sparse final layer ...")
        output_proj = train_sparse_final(
            final_layer,
            train_concept_loader,
            val_concept_loader,
            args.saga_n_iters,
            args.saga_lam,
            step_size=args.saga_step_size,
            device=args.device,
        )

    W_g = output_proj["path"][0]["weight"]
    b_g = output_proj["path"][0]["bias"]
    final_layer.load_state_dict({"weight": W_g, "bias": b_g})
    final_layer.save_model(save_dir)

    ##############################################
    #### Test the model on test set ####
    ##############################################
    test_accuracy = test_model(
        test_cbl_loader, backbone, cbl, normalization_layer, final_layer, args.device
    )
    logger.info(f"Test accuracy: {test_accuracy}")

    ##############################################
    # Store training metadata
    ##############################################
    with open(os.path.join(save_dir, "metrics.txt"), "w") as f:
        out_dict = {}
        out_dict["per_class_accuracies"] = per_class_accuracy(
            nn.Sequential(backbone, cbl, normalization_layer, final_layer).to(
                args.device
            ),
            test_cbl_loader,
            classes,
            device=args.device,
        )

        for key in ("lam", "lr", "alpha", "time"):
            out_dict[key] = float(output_proj["path"][0][key])
        out_dict["metrics"] = output_proj["path"][0]["metrics"]
        out_dict["metrics"]["test_accuracy"] = test_accuracy
        nnz = (W_g.abs() > 1e-5).sum().item()
        total = W_g.numel()
        out_dict["sparsity"] = {
            "Non-zero weights": nnz,
            "Total weights": total,
            "Percentage non-zero": nnz / total,
        }
        json.dump(out_dict, f, indent=2)

    utils.write_parameters_tensorboard(
        tb_writer, vars(args), test_accuracy * 100.0, (nnz / total) * 100.0
    )

    ##############################################
    ## Visualize top images for concepts ##
    ##############################################
    if args.visualize_concepts:
        target_layer = data_utils.BACKBONE_VISUALIZATION_TARGET_LAYER[args.backbone]
        os.mkdir(os.path.join(save_dir, "concept_visualization"))
        cbl_with_backbone = nn.Sequential(backbone, cbl).to(args.device)
        concepts_logits = []
        for (images_tensor, _, _) in tqdm(test_cbl_loader):
            images_tensor = images_tensor.to(args.device)
            with torch.no_grad():
                concepts_logits.append(
                    cbl_with_backbone(images_tensor).detach().cpu().numpy()
                )
        concepts_logits = np.concatenate(concepts_logits, axis=0)
        for concept_idx, concept in enumerate(concepts):
            fig = utils.display_top_activated_images(
                concept_idx,
                concepts_logits,
                cbl_with_backbone,
                target_layer,
                test_cbl_loader.dataset,
                transform=backbone.preprocess,
                device=args.device,
                k=10,
            )
            fig.savefig(
                os.path.join(save_dir, "concept_visualization", f"{concept}.png")
            )

    return save_dir


def main():
    sys.stdout = LoggerWriter("INFO")
    sys.stderr = LoggerWriter("DEBUG")

    parser = argparse.ArgumentParser(description="Settings for creating CBM")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument(
        "--concept_set", type=str, default=None, help="path to concept set name"
    )
    parser.add_argument(
        "--filter_set", type=str, default=None, help="path to concept set name"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split fraction"
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="clip_RN50",
        help="Which pretrained model to use as backbone",
    )
    parser.add_argument(
        "--feature_layer",
        type=str,
        default="layer4",
        help="Which layer to collect activations from. Should be the name of second to last layer in the model",
    )
    parser.add_argument(
        "--use_clip_penultimate",
        action="store_true",
        help="Whether to use the penultimate layer of the CLIP backbone",
    )
    parser.add_argument(
        "--skip_concept_filter",
        action="store_true",
        help="Whether to skip filtering concepts",
    )
    parser.add_argument(
        "--annotation_dir",
        type=str,
        default="outputs",
        help="where annotations are saved",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="saved_models",
        help="where to save trained models",
    )
    parser.add_argument(
        "--load_dir", type=str, default=None, help="where to load trained models from"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Which device to use"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers used for loading data",
    )
    parser.add_argument(
        "--allones_concept",
        action="store_true",
        help="Change concept dataset to ones corresponding to class",
    )
    # arguments for CBL
    parser.add_argument(
        "--crop_to_concept_prob",
        type=float,
        default=0.0,
        help="Probability of cropping to concept granuality",
    )
    parser.add_argument(
        "--cbl_confidence_threshold",
        type=float,
        default=0.15,
        help="Confidence threshold for bouding boxes to use",
    )
    parser.add_argument(
        "--cbl_hidden_layers",
        type=int,
        default=1,
        help="how many hidden layers to use in the projection layer",
    )
    parser.add_argument(
        "--cbl_batch_size",
        type=int,
        default=32,
        help="Batch size used when fitting projection layer",
    )
    parser.add_argument(
        "--cbl_epochs",
        type=int,
        default=20,
        help="how many steps to train the projection layer for",
    )
    parser.add_argument(
        "--cbl_weight_decay",
        type=float,
        default=1e-5,
        help="weight decay for training the projection layer",
    )
    parser.add_argument(
        "--cbl_lr",
        type=float,
        default=5e-4,
        help="learning rate for training the projection layer",
    )
    parser.add_argument(
        "--cbl_loss_type",
        choices=["bce", "twoway"],
        default="bce",
        help="Loss type for training CBL",
    )
    parser.add_argument(
        "--cbl_twoway_tp",
        type=float,
        default=4.0,
        help="TPE hyperparameter for TwoWay CBL loss",
    )
    parser.add_argument(
        "--cbl_pos_weight",
        type=float,
        default=1.0,
        help="loss weight for positive examples",
    )
    parser.add_argument(
        "--cbl_auto_weight",
        action="store_true",
        help="whether to automatically weight positive examples",
    )
    parser.add_argument(
        "--cbl_finetune",
        action="store_true",
        help="Enable finetuning backbone in CBL training",
    )
    parser.add_argument(
        "--cbl_bb_lr_rate",
        type=float,
        default=1,
        help="Rescale the learning rate of backbone during finetuning",
    )
    parser.add_argument(
        "--cbl_optimizer",
        choices=["adam", "sgd"],
        default="sgd",
        help="Optimizer used in CBL training.",
    )
    parser.add_argument(
        "--cbl_scheduler",
        choices=[None, "cosine"],
        default=None,
        help="Scheduler used in CBL training.",
    )
    # arguments for SAGA
    parser.add_argument(
        "--saga_batch_size",
        type=int,
        default=512,
        help="Batch size used when fitting final layer",
    )
    parser.add_argument(
        "--saga_step_size", type=float, default=0.1, help="Step size for SAGA"
    )
    parser.add_argument(
        "--saga_lam",
        type=float,
        default=0.0007,
        help="Sparsity regularization parameter, higher->more sparse",
    )
    parser.add_argument(
        "--saga_n_iters",
        type=int,
        default=2000,
        help="How many iterations to run the final layer solver for",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--dense", action="store_true", help="train with dense or sparse method"
    )
    parser.add_argument(
        "--dense_lr",
        type=float,
        default=0.001,
        help="Learning rate for the dense final layer training",
    )
    parser.add_argument("--data_parallel", action="store_true")
    parser.add_argument(
        "--visualize_concepts", action="store_true", help="Visualize concepts"
    )

    config_parser = argparse.ArgumentParser()
    config_parser.add_argument("--config", type=str, default=None)
    config_arg, remaining_args = config_parser.parse_known_args()
    if config_arg.config is not None:
        with open(config_arg.config, "r") as f:
            config_arg = json.load(f)
        parser.set_defaults(**config_arg)
    
    # run the training
    args = parser.parse_args(remaining_args)
    logger.info(args)
    
    # set random seed for reproducibility
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # train the model
    _ = train_cbm_and_save(args)


if __name__ == "__main__":
    main()
