import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import imread_any


DEFAULT_EXTENSIONS: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')


@dataclass(frozen=True)
class DatasetConfig:
    root_dir: str
    extensions: Tuple[str, ...] = DEFAULT_EXTENSIONS

class KnotImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions: Optional[Iterable[str]] = None):
        self.root = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        self.extensions = tuple(ext.lower() for ext in (extensions or DEFAULT_EXTENSIONS))
        classes = sorted(
            [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        )
        for i, c in enumerate(classes):
            self.class_to_idx[c] = i
            folder = os.path.join(root_dir, c)
            for fname in os.listdir(folder):
                if fname.lower().endswith(self.extensions):
                    self.samples.append((os.path.join(folder, fname), i, c))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, label_name = self.samples[idx]
        img = imread_any(path)
        img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)
        else:
            
            img = torch.from_numpy(np.array(img).transpose(2,0,1)).float()/255.0
        return img, label, path, label_name
