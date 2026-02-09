import numpy as np
import cv2
from skimage.morphology import skeletonize
from skimage.filters import threshold_otsu


def preprocess_for_skeleton(img_rgb, width=512):
    
    h, w = img_rgb.shape[:2]
    scale = width / float(w)
    img = cv2.resize(img_rgb, (width, int(h*scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    gray = cv2.equalizeHist(gray)
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 31, 10)
    if th.mean() < 10:
        t = threshold_otsu(blur)
        th = (blur < t).astype('uint8') * 255
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    
    sk = skeletonize(opened.astype(bool))
    return sk, gray
