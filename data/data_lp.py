import pickle
from pathlib import Path
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from torchvision.transforms import InterpolationMode

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def pickle_load(f_name):
    try:
        with open(f_name, 'rb') as f:
            return pickle.load(f)
    except:
        print(f_name)
        raise RuntimeError('cannot load file')


def pickle_dump(obj, f_name):
    with open(f_name, 'wb') as f:
        pickle.dump(obj, f)

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])


preprocess = _transform(224)


class LinearProbeDataset(Dataset):
    """
    Provide (image, label) pair for association matrix optimization,
    where image is a tensor representing image after transformation
    """

    def __init__(self, data_path, cls_names_file, split="train", img_ext='.jpg', transform=None):
        cls2img = pickle_load(
            f'{data_path}/splits/class2images_{split}.p'
        )
        self.data_path = f"{data_path}/images"
        self.images = []
        self.labels = []
        self.img_ext = img_ext
        self.transform = transform

        cls_names = []
        # cls_names_file file is text file with class names in each line
        with open(cls_names_file, 'r') as f:
            for line in f:
                cls_names.append(line.strip())
            print(f"Found {len(cls_names)} classes")

        for cls_name, imgs in cls2img.items():
            try:
                label = cls_names.index(cls_name)
            except:
                label_modifed = cls_name.replace(' ', '_')
                label = cls_names.index(label_modifed)
                print(f"Class name {cls_name} not found in class names file, using {label_modifed} instead")
            self.images.extend(imgs)
            self.labels += [label] * len(imgs)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # esnure all images are in RGB format
        image = Image.open(
                f"{self.data_path}/{self.images[idx]}{self.img_ext}"
        )
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


