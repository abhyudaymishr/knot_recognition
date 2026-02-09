import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import inspect
import numpy as np

try:
    from .utils import imread_any
except ImportError:
    from utils import imread_any

# Flexible import for cross-module usage
try:
    from .dataset import KnotImageDataset
except ImportError:
    from dataset import KnotImageDataset

class KnotImageDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, extensions=None):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        for i,c in enumerate(classes):
            self.class_to_idx[c]=i
            folder = os.path.join(root_dir, c)
            for fname in os.listdir(folder):
                if fname.lower().endswith(('.png','.jpg','.jpeg','.gif','.bmp','.tiff')):
                    self.samples.append((os.path.join(folder,fname), i, c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, label_name = self.samples[idx]
        img = imread_any(path)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            # default: convert to tensor
            img = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        return img, label, path, label_name


def get_resnet(num_classes=1000, pretrained=True, model_name="resnet18", freeze_backbone=False):
    
    ctor = getattr(models, model_name, None)
    if ctor is None:
        raise ValueError(f"Unknown model_name '{model_name}'. Expected torchvision resnet variant.")

    weights_map = {
        "resnet18": "ResNet18_Weights",
        "resnet34": "ResNet34_Weights",
        "resnet50": "ResNet50_Weights",
        "resnet101": "ResNet101_Weights",
        "resnet152": "ResNet152_Weights",
    }

    sig = inspect.signature(ctor)
    if "weights" in sig.parameters:
        weights_enum = getattr(models, weights_map.get(model_name, ""), None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum is not None) else None
        model = ctor(weights=weights)
    elif "pretrained" in sig.parameters:
        model = ctor(pretrained=pretrained)
    else:
        model = ctor()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith("fc.")

    return model
