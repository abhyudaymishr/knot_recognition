import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np

from .utils import imread_any

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
