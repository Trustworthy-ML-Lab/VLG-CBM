import json
import os
from functools import partial
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm

import model.utils as utils
import data.utils as data_utils
from model.cbm import Backbone, ConceptLayer, NormalizationLayer
from data.utils import format_concept, get_classes
from glm_saga.elasticnet import IndexedTensorDataset
from data.utils import plot_annotations


class ConceptDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        torch_dataset: Dataset,
        concepts: List[str] = None,
        split_suffix="train",
        label_dir: str = "outputs",
        confidence_threshold: float = 0.10,
        preprocess=None,
        crop_to_concept_prob: bool = 0.0,
        overlap_iou_threshold: float = 0.5,
        concept_only=False
    ):
        self.torch_dataset = torch_dataset
        self.concepts = concepts
        self.dir = f"{label_dir}/{dataset_name}_{split_suffix}"
        self.confidence_threshold = confidence_threshold
        self.preprocess = preprocess
        self.overlap_iou_threshold = overlap_iou_threshold
        self.concept_only = concept_only
        # Return cropped image containing a single concept
        # with probability `crop_to_concept_prob`
        self.crop_to_concept_prob = crop_to_concept_prob

    def __len__(self):
        return len(self.torch_dataset)

    def __getitem__(self, idx):
        if self.concept_only:
            return 0, self._get_concept(idx), 0 # 0 is placeholder
        prob = np.random.rand()
        if prob < self.crop_to_concept_prob:
            try:
                return self.__getitem__per_concept(idx)
            except Exception as e:
                logger.warning(f"Failed to get item {idx} per concept: {e}")

        return self.__getitem__all(idx)

    def __getitem__per_concept(self, idx):
        image, target = self.torch_dataset[idx]

        # return 1 hot vector of concepts
        data = self._get_data(idx)

        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])

        # get mapping of concepts to a random bounding box containing the concept
        concept_bbx_map = []
        for concept_idx, concept in enumerate(self.concepts):
            _, matched_bbxs = self._find_in_list(concept, bbxs)
            if len(matched_bbxs) > 0:
                concept_bbx_map.append((concept_idx, matched_bbxs[np.random.randint(0, len(matched_bbxs))]))

        # get one hot vector of concepts
        concept_one_hot = torch.zeros(len(self.concepts), dtype=torch.float)
        if len(concept_bbx_map) > 0:
            # randomly pick a concept and its bounding box
            random_concept_idx, random_bbx = concept_bbx_map[np.random.randint(0, len(concept_bbx_map))]
            concept_one_hot[random_concept_idx] = 1.0
            image = image.crop(random_bbx["box"])

            # mark concepts with high overlap with the selected concept as 1
            for bbx in bbxs:
                if bbx["label"] == random_bbx["label"]:
                    continue
                else:
                    iou = utils.get_bbox_iou(random_bbx["box"], bbx["box"])
                    try:
                        if iou > self.overlap_iou_threshold:
                            concept_idx = self.concepts.index(bbx["label"])
                            concept_one_hot[concept_idx] = 1.0
                            # logger.debug(f"Marking {bbx['concept']} as 1 due to overlap with {random_bbx['concept']}")
                    except ValueError:
                        continue

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target

    def __getitem__all(self, idx):
        image, target = self.torch_dataset[idx]

        # get raw data
        data = self._get_data(idx)

        # get one hot vector of concepts
        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])

        # get one hot vector of concepts
        concept_one_hot = [1 if self._find_in_list(concept, bbxs)[0] else 0 for concept in self.concepts]
        concept_one_hot = torch.tensor(concept_one_hot, dtype=torch.float)

        # preprocess image
        if self.preprocess:
            image = self.preprocess(image)

        return image, concept_one_hot, target
    
    def _get_concept(self, idx):
        # return 1 hot vector of concepts
        data = self._get_data(idx)

        # get one hot vector of concepts
        bbxs = data[1:]
        bbxs = [bbx for bbx in bbxs if bbx["logit"] > self.confidence_threshold]
        for bbx in bbxs:
            bbx["label"] = format_concept(bbx["label"])

        # get one hot vector of concepts
        concept_one_hot = [1 if self._find_in_list(concept, bbxs)[0] else 0 for concept in self.concepts]
        concept_one_hot = torch.tensor(concept_one_hot, dtype=torch.float)
        return concept_one_hot

    def _find_in_list(self, concept: str, bbxs: List[Dict[str, Any]]) -> Tuple[bool, List[Dict[str, Any]]]:
        # randomly pick a bounding box
        matched_bbxs = [bbx for bbx in bbxs if concept == bbx["label"]]
        return len(matched_bbxs) > 0, matched_bbxs

    def _get_data(self, idx):
        data_file = f"{self.dir}/{idx}.json"
        with open(data_file, "r") as f:
            data = json.load(f)
        return data

    def get_annotations(self, idx: int):
        return self._get_data(idx)[1:]

    def visualize_annotations(self, idx: int):
        image_pil = self.torch_dataset[idx][0]
        annotations = self._get_data(idx)[1:]
        fig = plot_annotations(image_pil, annotations)
        fig.show()

    def plot_annotations(self, idx: int, annotations: List[Dict[str, Any]]):
        image_pil = self.torch_dataset[idx][0]
        fig = plot_annotations(image_pil, annotations)
        fig.show()

    def get_image_pil(self, idx: int):
        return self.torch_dataset[idx][0]

    def get_target(self, idx):
        _, target = self.torch_dataset[idx]
        return target    

class AllOneConceptDataset(ConceptDataset):
    def __init__(self, classes, *args, **kwargs):
        print(args, kwargs)
        super().__init__(*args, **kwargs)
        self.per_class_concepts = len(self.concepts) // len(classes)
        logger.info(f"Assigning {self.per_class_concepts} concepts to each class")

    def __getitem__(self, idx):
        image, target = self.torch_dataset[idx]
        if self.preprocess:
            image = self.preprocess(image)
        concept_one_hot = torch.zeros((len(self.concepts),), dtype=torch.float)
        concept_one_hot[target * self.per_class_concepts : (target + 1) * self.per_class_concepts] = 1
        return image, concept_one_hot, target


def get_concept_dataloader(
    dataset_name: str,
    split: str,
    concepts: List[str],
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    shuffle: bool = False,
    confidence_threshold: float = 0.10,
    crop_to_concept_prob: float = 0.0,
    label_dir="outputs",
    use_allones=False,
    seed: int = 42,
    concept_only=False
):
    dataset = ConceptDataset if not use_allones else partial(AllOneConceptDataset, get_classes(dataset_name))
    if split == "test":
        dataset = dataset(
            dataset_name,
            data_utils.get_data(f"{dataset_name}_val", None),
            concepts,
            split_suffix="val",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )
        logger.info(f"Test dataset size: {len(dataset)}")
    else:
        assert val_split is not None
        dataset = dataset(
            dataset_name,
            data_utils.get_data(f"{dataset_name}_train", None),
            concepts,
            split_suffix="train",
            preprocess=preprocess,
            confidence_threshold=confidence_threshold,
            crop_to_concept_prob=crop_to_concept_prob,
            label_dir=label_dir,
            concept_only=concept_only
        )

        # get split indices
        n_val = int(val_split * len(dataset))
        n_train = len(dataset) - n_val
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(seed)
        )  # ensure same split in same run

        if split == "train":
            logger.info(f"Train dataset size: {len(train_dataset)}")
            dataset = train_dataset
        else:
            logger.info(f"Val dataset size: {len(val_dataset)}")
            dataset = val_dataset

    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return loader


def get_filtered_concepts_and_counts(
    dataset_name,
    raw_concepts,
    preprocess=None,
    val_split: Optional[float] = 0.1,
    batch_size: int = 256,
    num_workers: int = 4,
    confidence_threshold: float = 0.10,
    label_dir="outputs",
    use_allones: bool = False,
    seed: int = 42,
):
    # remove concepts that are not present in the dataset
    dataloader = get_concept_dataloader(
        dataset_name,
        "train",
        raw_concepts,
        preprocess=preprocess,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        confidence_threshold=confidence_threshold,
        crop_to_concept_prob=0.0,
        label_dir=label_dir,
        use_allones=use_allones,
        seed=seed,
        concept_only=True
    )
    # get concept counts
    raw_concepts_count = torch.zeros(len(raw_concepts))
    for data in tqdm(dataloader):
        raw_concepts_count += data[1].sum(dim=0)

    # remove concepts that are not present in the dataset
    raw_concepts_count = raw_concepts_count.numpy()
    concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count > 0]
    concept_counts = [count for _, count in zip(raw_concepts, raw_concepts_count) if count > 0]
    filtered_concepts = [concept for concept, count in zip(raw_concepts, raw_concepts_count) if count == 0]
    print(f"Filtered {len(raw_concepts) - len(concepts)} concepts")

    return concepts, concept_counts, filtered_concepts


def get_final_layer_dataset(
    backbone: Backbone,
    cbl: ConceptLayer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    save_dir: str,
    load_dir: str = None,
    batch_size: int = 256,
    device="cuda",
    filter=None,
):
    if load_dir is None:
        logger.info("Creating final layer training and validation datasets")
        with torch.no_grad():
            train_concept_features = []
            train_concept_labels = []
            logger.info("Creating final layer training dataset")
            for features, _, labels in tqdm(train_loader):
                features = features.to(device)
                concept_logits = cbl(backbone(features))
                train_concept_features.append(concept_logits.detach().cpu())
                train_concept_labels.append(labels)
            train_concept_features = torch.cat(train_concept_features, dim=0)
            train_concept_labels = torch.cat(train_concept_labels, dim=0)

            val_concept_features = []
            val_concept_labels = []
            logger.info("Creating final layer validation dataset")
            for features, _, labels in tqdm(val_loader):
                features = features.to(device)
                concept_logits = cbl(backbone(features))
                val_concept_features.append(concept_logits.detach().cpu())
                val_concept_labels.append(labels)
            val_concept_features = torch.cat(val_concept_features, dim=0)
            val_concept_labels = torch.cat(val_concept_labels, dim=0)

            # normalize concept features
            train_concept_features_mean = train_concept_features.mean(dim=0)
            train_concept_features_std = train_concept_features.std(dim=0)
            train_concept_features = (train_concept_features - train_concept_features_mean) / train_concept_features_std
            val_concept_features = (val_concept_features - train_concept_features_mean) / train_concept_features_std

            # normalization layer
            normalization_layer = NormalizationLayer(train_concept_features_mean, train_concept_features_std, device=device)
    else:
        # load normalized concept features
        logger.info("Loading final layer training dataset")
        train_concept_features = torch.load(os.path.join(load_dir, "train_concept_features.pt"))
        train_concept_labels = torch.load(os.path.join(load_dir, "train_concept_labels.pt"))
        val_concept_features = torch.load(os.path.join(load_dir, "val_concept_features.pt"))
        val_concept_labels = torch.load(os.path.join(load_dir, "val_concept_labels.pt"))
        normalization_layer = NormalizationLayer.from_pretrained(load_dir, device=device)

    # save normalized concept features
    torch.save(train_concept_features, os.path.join(save_dir, "train_concept_features.pt"))
    torch.save(train_concept_labels, os.path.join(save_dir, "train_concept_labels.pt"))
    torch.save(val_concept_features, os.path.join(save_dir, "val_concept_features.pt"))
    torch.save(val_concept_labels, os.path.join(save_dir, "val_concept_labels.pt"))

    # save normalized concept features mean and std
    normalization_layer.save_model(save_dir)
    if filter is not None:
        train_concept_features = train_concept_features[:, filter]
        val_concept_features = val_concept_features[:, filter]
    # Note: glm saga expects y to be on CPU
    train_concept_dataset = IndexedTensorDataset(train_concept_features, train_concept_labels)
    val_concept_dataset = TensorDataset(val_concept_features, val_concept_labels)
    logger.info("Train concept dataset size: {}".format(len(train_concept_dataset)))
    logger.info("Val concept dataset size: {}".format(len(val_concept_dataset)))

    train_concept_loader = DataLoader(train_concept_dataset, batch_size=batch_size, shuffle=True)
    val_concept_loader = DataLoader(val_concept_dataset, batch_size=batch_size, shuffle=False)
    return train_concept_loader, val_concept_loader, normalization_layer
