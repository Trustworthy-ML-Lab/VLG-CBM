import os
import math
import shutil
import torch
import clip
from data import utils as data_utils
import json
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from interpretability.cam import ScoreCAM
from interpretability.visualize import visualize

PM_SUFFIX = {"max": "_max", "avg": ""}


def save_target_activations(
    target_model, dataset, save_name, target_layers=["layer4"], batch_size=1000, device="cuda", pool_mode="avg"
):
    """
    save_name: save_file path, should include {} which will be formatted by layer names
    """
    _make_save_dir(save_name)
    save_names = {}
    for target_layer in target_layers:
        save_names[target_layer] = save_name.format(target_layer)

    if _all_saved(save_names):
        return

    all_features = {target_layer: [] for target_layer in target_layers}

    hooks = {}
    for target_layer in target_layers:
        command = "target_model.{}.register_forward_hook(get_activation(all_features[target_layer], pool_mode))".format(
            target_layer
        )
        hooks[target_layer] = eval(command)

    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = target_model(images.to(device))

    for target_layer in target_layers:
        torch.save(torch.cat(all_features[target_layer]), save_names[target_layer])
        hooks[target_layer].remove()
    # free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_image_features(model, dataset, save_name, batch_size=1000, device="cuda"):
    _make_save_dir(save_name)
    all_features = []

    if os.path.exists(save_name):
        return

    save_dir = save_name[: save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=8, pin_memory=True)):
            features = model.encode_image(images.to(device))
            all_features.append(features.cpu())
    torch.save(torch.cat(all_features), save_name)
    # free memory
    del all_features
    torch.cuda.empty_cache()
    return


def save_clip_text_features(model, text, save_name, batch_size=1000):
    if os.path.exists(save_name):
        return
    _make_save_dir(save_name)
    text_features = []
    with torch.no_grad():
        for i in tqdm(range(math.ceil(len(text) / batch_size))):
            text_features.append(model.encode_text(text[batch_size * i : batch_size * (i + 1)]))
    text_features = torch.cat(text_features, dim=0)
    torch.save(text_features, save_name)
    del text_features
    torch.cuda.empty_cache()
    return


def save_activations(clip_name, target_name, target_layers, d_probe, concept_set, batch_size, device, pool_mode, save_dir):
    target_save_name, clip_save_name, text_save_name = get_save_names(
        clip_name, target_name, "{}", d_probe, concept_set, pool_mode, save_dir
    )
    save_names = {"clip": clip_save_name, "text": text_save_name}
    for target_layer in target_layers:
        save_names[target_layer] = target_save_name.format(target_layer)

    if _all_saved(save_names):
        return

    clip_model, clip_preprocess = clip.load(clip_name, device=device)

    if target_name.startswith("clip_"):
        target_model, target_preprocess = clip.load(target_name[5:], device=device)
    else:
        target_model, target_preprocess = data_utils.get_target_model(target_name, device)
    # setup data
    data_c = data_utils.get_data(d_probe, clip_preprocess)
    data_t = data_utils.get_data(d_probe, target_preprocess)

    with open(concept_set, "r") as f:
        words = (f.read()).split("\n")
    text = clip.tokenize(["{}".format(word) for word in words]).to(device)

    save_clip_text_features(clip_model, text, text_save_name, batch_size)

    save_clip_image_features(clip_model, data_c, clip_save_name, batch_size, device)
    if target_name.startswith("clip_"):
        save_clip_image_features(target_model, data_t, target_save_name, batch_size, device)
    else:
        save_target_activations(target_model, data_t, target_save_name, target_layers, batch_size, device, pool_mode)

    return


def get_similarity_from_activations(target_save_name, clip_save_name, text_save_name, similarity_fn, return_target_feats=True):
    image_features = torch.load(clip_save_name)
    text_features = torch.load(text_save_name)
    with torch.no_grad():
        image_features /= image_features.norm(dim=-1, keepdim=True).float()
        text_features /= text_features.norm(dim=-1, keepdim=True).float()
        clip_feats = image_features @ text_features.T
    del image_features, text_features
    torch.cuda.empty_cache()

    target_feats = torch.load(target_save_name)
    similarity = similarity_fn(clip_feats, target_feats)

    del clip_feats
    torch.cuda.empty_cache()

    if return_target_feats:
        return similarity, target_feats
    else:
        del target_feats
        torch.cuda.empty_cache()
        return similarity


def get_activation(outputs, mode):
    """
    mode: how to pool activations: one of avg, max
    for fc neurons does no pooling
    """
    if mode == "avg":

        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.mean(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())

    elif mode == "max":

        def hook(model, input, output):
            if len(output.shape) == 4:
                outputs.append(output.amax(dim=[2, 3]).detach().cpu())
            elif len(output.shape) == 2:
                outputs.append(output.detach().cpu())

    return hook


def get_save_names(clip_name, target_name, target_layer, d_probe, concept_set, pool_mode, save_dir):
    if target_name.startswith("clip_"):
        target_save_name = "{}/{}_{}.pt".format(save_dir, d_probe, target_name.replace("/", ""))
    else:
        target_save_name = "{}/{}_{}_{}{}.pt".format(save_dir, d_probe, target_name, target_layer, PM_SUFFIX[pool_mode])
    clip_save_name = "{}/{}_clip_{}.pt".format(save_dir, d_probe, clip_name.replace("/", ""))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace("/", ""))

    return target_save_name, clip_save_name, text_save_name


def _all_saved(save_names):
    """
    save_names: {layer_name:save_path} dict
    Returns True if there is a file corresponding to each one of the values in save_names,
    else Returns False
    """
    for save_name in save_names.values():
        if not os.path.exists(save_name):
            return False
    return True


def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[: save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def get_accuracy_cbm(model, dataset, device, batch_size=250, num_workers=2):
    correct = 0
    total = 0
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        with torch.no_grad():
            # outs = target_model(images.to(device))
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            correct += torch.sum(pred.cpu() == labels)
            total += len(labels)
    return correct / total


def get_preds_cbm(model, dataset, device, batch_size=250, num_workers=2):
    preds = []
    for images, labels in tqdm(DataLoader(dataset, batch_size, num_workers=num_workers, pin_memory=True)):
        with torch.no_grad():
            outs, _ = model(images.to(device))
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def get_concept_act_by_pred(model, dataset, device):
    preds = []
    concept_acts = []
    for images, labels in tqdm(DataLoader(dataset, 500, num_workers=8, pin_memory=True)):
        with torch.no_grad():
            outs, concept_act = model(images.to(device))
            concept_acts.append(concept_act.cpu())
            pred = torch.argmax(outs, dim=1)
            preds.append(pred.cpu())
    preds = torch.cat(preds, dim=0)
    concept_acts = torch.cat(concept_acts, dim=0)
    concept_acts_by_pred = []
    for i in range(torch.max(pred) + 1):
        concept_acts_by_pred.append(torch.mean(concept_acts[preds == i], dim=0))
    concept_acts_by_pred = torch.stack(concept_acts_by_pred, dim=0)
    return concept_acts_by_pred


def weight_truncation(weight: torch.Tensor, sparsity: float):
    numel = weight.numel()
    num_zeros = int((1 - sparsity) * numel)
    threshold = torch.sort(weight.flatten().abs())[0][num_zeros]
    sparse_weight = weight.clone().detach()
    sparse_weight[weight.abs() < threshold] = 0
    return sparse_weight


def write_parameters_tensorboard(
    tb_writer: SummaryWriter, parameter_dict, test_accuracy, sparsity
):
    tb_writer.add_hparams(
        parameter_dict,
        {
            "Test accuracy": test_accuracy,
            "sparsity": sparsity,
        },
        run_name="default",
    )


def update_tensorboard_dir(dir):
    try:
        # load args dir
        with open(f"{dir}/args.txt", "r") as f:
            args = json.load(f)

        # load metrics
        with open(f"{dir}/metrics.txt", "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(e)
        return

    # delete all folders in the directory
    # Iterate over all items in the directory
    for item in os.listdir(dir):
        item_path = os.path.join(dir, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            # If the item is a directory, delete it
            shutil.rmtree(item_path)

    tb_writer = SummaryWriter(dir)
    test_accuracy = float(metrics["metrics"]["test_accuracy"]) * 100.0
    sparsity = float(metrics["sparsity"]["Percentage non-zero"]) * 100.0

    write_parameters_tensorboard(tb_writer, args, test_accuracy, sparsity)


def update_tensorboard_dirs(save_dir):
    for dir in os.listdir(save_dir):
        dir_path = os.path.join(save_dir, dir)
        if os.path.isdir(dir_path):
            logger.info(f"Updating {dir_path}")
            update_tensorboard_dir(dir_path)

def get_per_class_filtered_concepts(dataset, save_path=None):
    classes = data_utils.get_classes(dataset)
    files = [
        f"data/concept_sets/gpt3_init/gpt3_{dataset}_around.json",
        f"data/concept_sets/gpt3_init/gpt3_{dataset}_important.json",
        f"data/concept_sets/gpt3_init/gpt3_{dataset}_superclass.json",
    ]

    # load raw concepts per class
    raw_concepts_per_class = {}
    for file in files:
        with open(file, "r") as f:
            _original_concepts_per_class = json.load(f)
            for key in _original_concepts_per_class.keys():
                if key in raw_concepts_per_class.keys():
                    raw_concepts_per_class[key].extend(
                        [data_utils.format_concept(m) for m in _original_concepts_per_class[key]]
                    )
                else:
                    raw_concepts_per_class[key] = [data_utils.format_concept(m) for m in  _original_concepts_per_class[key]]

    # load filtered concepts
    filtered_concepts = data_utils.get_concepts(concept_file=f"data/concept_sets/{dataset}_filtered.txt")

    # map filtered concepts to class
    filtered_concepts_per_class = {}
    for class_idx in range(len(classes)):
        filtered_concepts_per_class[classes[class_idx]] = [
            filtered_concept
            for filtered_concept in filtered_concepts
            if filtered_concept in raw_concepts_per_class[classes[class_idx]]
        ]

    # save filtered concepts per class
    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(filtered_concepts_per_class, f)

    return raw_concepts_per_class

def get_bbox_iou(boxA, boxB):
    # Source: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
 

def display_top_activated_images(concept_idx, concepts_logits, model, target_layer, dataset, transform, k=20, device="cuda"):
    logger.info(f"Display top {k} activated images for concept idx: {concept_idx}")
    concept_logit = concepts_logits[:, concept_idx]

    # sort by descending order of activation
    sorted_idx = np.argsort(-concept_logit)
    top_k_sorted_idx = sorted_idx[:k]

    # is instance of torch.utils.data.Subset
    is_subset = False
    if isinstance(dataset, torch.utils.data.Subset):
        is_subset = True
        logger.info(f"Dataset is a subset of dataset: {dataset}")

    # setup target layer
    # model[0].backbone.features.stage4.unit2.body.conv2.conv
    ig = ScoreCAM(model, eval(f"model[0].backbone.{target_layer}"))

    # setup transforms
    transform_without_norm = transforms.Compose(transform.transforms[:-1])
    transform_normalize = transforms.Compose(transform.transforms[-1:])
    logger.info(f"Original transform: {transform}")
    logger.info(f"Transform without normalization: {transform_without_norm}")
    logger.info(f"Transform normalization: {transform_normalize}")

    # create a figure to display top k images
    plt.ioff()
    fig, ax = plt.subplots(1, k, figsize=(20, 5))
    
    # visualize top k images and their attributions
    for i, idx in tqdm(enumerate(top_k_sorted_idx)):
        if is_subset:
            logger.info(f"Subset dataset idx: {idx}, Score: {concept_logit[idx]}, Actual class: {dataset.dataset.get_target(dataset.indices[idx])}")
            rgb_image = dataset.dataset.get_image_pil(dataset.indices[idx])
        else:
            logger.info(f"Dataset idx: {idx}, Score: {concept_logit[idx]}, Actual class: {dataset.get_target(idx)}")
            rgb_image = dataset.get_image_pil(idx)

        # normalize image
        unnormalized_image = transform_without_norm(rgb_image)
        normalized_image = transform_normalize(unnormalized_image)

        # display attributions
        cam, pred_idx = ig(normalized_image.unsqueeze(0).to(device), idx=concept_idx)
        logger.info(f"Predicted concept: {pred_idx}")
        heatmap = visualize(unnormalized_image.unsqueeze(0), cam)
        embedded_heatmap = Image.fromarray(np.uint8(heatmap.squeeze().numpy().transpose(1, 2, 0) * 255))
        ax[i].imshow(embedded_heatmap)

        # turn off axis
        ax[i].axis('off')

    # add concept idx to the figure with less padding
    fig.suptitle(f"Concept idx: {concept_idx}")
    fig.tight_layout()

    return fig


def get_bbox_iou(boxA, boxB):
    # Source: https://gist.github.com/meyerjo/dd3533edc97c81258898f60d8978eddc
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou
 
def rowwise_truncation(weight: torch.Tensor, sparsity: float):
    numel = weight.numel()
    num_zeros = int((1 - sparsity) * numel)
    sparse_weight = weight.clone().detach()
    row_number = round(weight.size(1) * sparsity)
    _, indices = sparse_weight.topk(row_number, dim=1)
    results = torch.zeros_like(sparse_weight)
    for row in range(results.size(0)):
        results[row, indices[row]] = sparse_weight[row, indices[row]]
    return results