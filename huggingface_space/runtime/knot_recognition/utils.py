import os
import cv2
import numpy as np
from PIL import Image


def imread_any(path):
    im = Image.open(path).convert('RGB')
    return np.array(im)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)