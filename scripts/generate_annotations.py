import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import data.utils as utils
import GroundingDINO.groundingdino.datasets.transforms as T
from data.utils import get_data, plot_annotations
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class Resize(object):
    def __init__(self, size):
        self.size = size
        self.resize = transforms.Resize((size, size))

    def __call__(self, img, target):
        return self.resize(img), target


def load_annotation_model(model_config_path, model_checkpoint_path, device="cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    logger.info(load_res)
    model.eval()
    model.to(device)
    tokenlizer = model.tokenizer
    return model, tokenlizer


def get_predictions(model: Any, images_tensor: torch.Tensor, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get predictions for a batch of images

    Args:
        model (Any): The model used for predictions.
        images_tensor (torch.Tensor): The images tensor.
        prompts (List[str]): The prompts associated with the images.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The logits and bounding boxes.
    """

    with torch.no_grad():
        outputs = model(images_tensor, captions=prompts)

    # get output
    logits = outputs["pred_logits"].sigmoid()  # (batch_size, nq, 256)
    boxes = outputs["pred_boxes"]  # (batch_size, nq, 4)

    return logits, boxes


def process_annotations_for_bbox(
    prompt: str,
    bbox: np.array,
    prompt_logits: np.array,
    tokenlizer: Any,
    text_threshold: float,
) -> List[Dict]:
    """
    Find concepts associated with a bounding box

    Args:
        prompt (str): The prompt or label associated associated with the prediction.
        bbox (np.ndarray): The bounding box coordinates as a numpy array.
        prompt_logit (np.ndarray): The logits over each token of prompt.
        tokenlizer (Any): The tokenizer used for processing the prompt.
        text_threshold (float): The threshold value used for text processing. Any concept below the threshold
        is ignored.

    Returns:
        List[Dict]: A list of dictionaries containing the following keys:
        - logits: The logits associated with each token of the concept.
        - score: The perplexity of the concept.
        - concept: The concept associated with the bounding box.
        - bbox: The bounding box coordinates.
    """

    assert len(prompt_logits.shape) == 1
    prompt_logits = prompt_logits[1:-1]  # Remove start and end token
    prompt_tokenized = tokenlizer(prompt)
    prompt_tokenized = prompt_tokenized["input_ids"][1:-1]

    # 1012 is the split token corresponding to "."
    split_token_idxs = [i for i, x in enumerate(prompt_tokenized) if x == 1012]

    # split the prompt into concepts and calculate perplexity
    annotations = []
    start = 0
    for split_index in split_token_idxs:
        # get the text and corresponding logits
        concept_tokenized = prompt_tokenized[start:split_index]
        concept_logits = prompt_logits[start:split_index]
        concept_probs = np.prod(concept_logits)

        # calculate perplexity
        concept_perplexity = concept_probs ** (1 / len(concept_tokenized))
        if concept_perplexity > text_threshold:
            annotations.append(
                {
                    "logits": concept_logits,
                    "logit": concept_perplexity,
                    "label": tokenlizer.decode(concept_tokenized),
                    "box": bbox,
                }
            )

        # update start index
        start = split_index + 1

    return annotations


def process_annotations(
    image_pil: Image.Image, prompt: str, logits: np.array, boxes: np.array, tokenlizer: Any, text_threshold: float
) -> List[Dict]:
    """
    Obtain annotations for a single image
    Args:
        image_pil (Image.Image): The PIL image
        prompt (str): The prompt associated associated with the prediction.
        logits (np.array): The logits associated with the predictions.
        boxes (np.array): The bounding box associated with the predictions.
        tokenlizer (Any): The tokenizer used for processing the prompt.
        text_threshold (float): The threshold value used for text processing. Any concept below the threshold
        is ignored.

    Returns:
        List[Dict]: A list of dictionaries containing the following
        keys:
        - logits: The logits associated with each token of the concept.
        - score: The perplexity of the concept.
        - concept: The concept associated with the bounding box.
        - bbox: The bounding box coordinates.
    """
    annotations = []
    for logit, bbox in zip(logits, boxes):
        _annotations = process_annotations_for_bbox(prompt, bbox, logit, tokenlizer, text_threshold)
        annotations.extend(_annotations)

    image = np.array(image_pil)
    H, W = image.shape[:2]
    for annotation in annotations:
        bbox = annotation["box"] * np.array([W, H, W, H])
        bbox[:2] -= bbox[2:] / 2
        bbox[2:] += bbox[:2]
        annotation["box"] = bbox

    return annotations

def save_annotations(
    output_dir: str, file_name: str, annotations: List[Union[str, dict]], img_path: Optional[str] = None
) -> None:
    """
    Save annotations to a json file

    Args:
        output_dir (str): The output directory
        file_name (str): The file name
        annotations (List[Union[str, dict]]): The annotations to save in the following format:
            - logits: The logits associated with each token of the concept.
            - score: The perplexity of the concept.
            - concept: The concept associated with the bounding box.
            - bbox: The bounding box coordinates.
        img_path (Optional[str]): The image path.

    Returns:
        None
    """
    json_data = [
        {
            "img_path": img_path,
        }
    ]
    for annotation in annotations:
        # convert any ndarray to list
        for key, value in annotation.items():
            if isinstance(value, np.ndarray):
                annotation[key] = value.tolist()
            elif isinstance(value, float) or isinstance(value, np.float32):
                annotation[key] = float(value)
    
        # add annotation to json
        json_data.append(annotation)

    # save json
    with open(os.path.join(output_dir, file_name), "w") as f:
        json.dump(json_data, f)


def main():
    parser = argparse.ArgumentParser("Grounded-Segment-Anything Demo", add_help=True)
    parser.add_argument(
        "--per_class_concept_file",
        type=str,
        default=None,
        help="path to per class concept file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
        help="path to config file",
    )
    parser.add_argument(
        "--grounded_checkpoint",
        type=str,
        default="GroundingDINO/groundingdino_swinb_cogcoor.pth",
        help="path to checkpoint file",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="outputs",
        help="output directory",
    )
    parser.add_argument("--text_threshold", type=float, default=0.25, help="text threshold")
    parser.add_argument("--start_class_idx", type=int, default=None, help="start index (inclusive)")
    parser.add_argument("--end_class_idx", type=int, default=None, help="end index (exclusive)")
    parser.add_argument("--batch_size", type=int, default=48, help="batch size")
    parser.add_argument("--num_workers", type=int, default=8, help="num workers")
    parser.add_argument("--dataset", type=str, default="cifar10_train", help="dataset name")
    parser.add_argument("--device", type=str, default="cuda", help="running on cpu only!, default=False")
    parser.add_argument("--save_image", action="store_true", help="save image")
    args = parser.parse_args()

    # cfg
    config_file = args.config  # change the path of the model config file
    grounded_checkpoint = args.grounded_checkpoint  # change the path of the model
    text_threshold = args.text_threshold
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_name = args.dataset
    device = args.device
    save_image = args.save_image
    output_dir = f"{args.output_dir}/{dataset_name}"

    # load classes
    cls_file = utils.LABEL_FILES[args.dataset.split("_")[0]]
    with open(cls_file, "r") as f:
        classes = f.read().split("\n")

    # load per class concepts
    if args.per_class_concept_file is not None:
        per_class_concepts_file = args.per_class_concept_file
    else:
        per_class_concepts_file = f"concept_files/{dataset_name.split('_')[0]}_per_class.json"

    print(f"Loading per class concepts from: {per_class_concepts_file}")
    with open(per_class_concepts_file, "r") as f:
        per_class_concepts = json.load(f)

    # load PIL dataset to obtain original images
    pil_data = utils.get_data(dataset_name, preprocess=None)

    # make output annotation directory
    os.makedirs(output_dir, exist_ok=True)

    # prepare dataset
    transform = T.Compose(
        [
            Resize(800),
            T.RandomResize([800]),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = get_data(dataset_name, preprocess=lambda x: transform(x, None)[0])

    # load model
    model, tokenlizer = load_annotation_model(config_file, grounded_checkpoint, device=device)

    # setup prompt
    for class_idx, class_name in enumerate(classes):
        # check if class_idx is in range of start and end
        if args.start_class_idx is not None and class_idx < args.start_class_idx:
            continue
        if args.end_class_idx is not None and class_idx >= args.end_class_idx:
            continue

        print(f"Running on class Index: {class_idx}, class name: {class_name}")
        per_class_concept = per_class_concepts[class_name]

        # setup prompt
        # add class name since it leads to better bounding boxes
        # we remove it during model training in the Concept dataset class
        prompt = utils.format_concept(class_name) + " . "
        for concept in per_class_concept:
            prompt = prompt + f"{utils.format_concept(concept)} . "
        prompt = prompt.strip()
        print(f"Prompt for class {class_name}: {prompt}")

        # only load images with class_idx
        dataset_subset = torch.utils.data.Subset(dataset, np.where(np.array(dataset.targets) == class_idx)[0])
        dataloader = torch.utils.data.DataLoader(
            dataset_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )
        print(f"Number of images in class {class_name}: {len(dataset_subset)}")

        # run model
        for batch_idx, (images, _) in enumerate(tqdm(dataloader)):
            images = images.to(device)

            # get predictions for images
            logits, boxes = get_predictions(model, images, [prompt] * images.shape[0])

            # get annotations per-image
            for image_idx in range(logits.shape[0]):
                # get global image index
                processed_images = batch_idx * batch_size + image_idx
                global_idx = dataset_subset.indices[processed_images]
                image_pil = pil_data[global_idx][0]

                # process annotations
                logits_image = logits[image_idx].clone().cpu().numpy()
                bboxes_image = boxes[image_idx].clone().cpu().numpy()
                annotations = process_annotations(image_pil, prompt, logits_image, bboxes_image, tokenlizer, text_threshold)

                # plot annotations on image
                if save_image:
                    fig = plot_annotations(image_pil, annotations)
                    plt.savefig(
                        os.path.join(output_dir, f"{global_idx}.jpg"),
                        bbox_inches="tight",
                        dpi=300,
                        pad_inches=0.0,
                    )
                    plt.close(fig)

                # set image path
                if dataset_name == "places365_val" or dataset_name == "places365_train":
                    img_path = dataset.imgs[global_idx]
                else:
                    img_path = None

                # save annotations
                save_annotations(
                    output_dir,
                    f"{global_idx}.json",
                    annotations,
                    img_path,
                )


if __name__ == "__main__":
    main()
