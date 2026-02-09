
import numpy as np
import cv2
from skimage.color import rgb2gray
from skimage.transform import resize
from scipy.fft import fft


def hu_moments(img_gray):
    # expects uint8 grayscale
    moments = cv2.moments(img_gray)
    hu = cv2.HuMoments(moments).flatten()
    # log transform to reduce dynamic range
    for i in range(len(hu)):
        if hu[i]==0:
            hu[i]=0.0
        else:
            hu[i] = -np.sign(hu[i]) * np.log10(abs(hu[i]) + 1e-30)
    return hu


def radial_fourier_descriptor(img_gray, n_bins=128, nf=16):
    # compute centroid, sample radial profile (averaged over angles)
    H,W = img_gray.shape
    yy, xx = np.mgrid[0:H,0:W]
    cx = W/2
    cy = H/2
    rr = np.sqrt((xx-cx)**2 + (yy-cy)**2)
    rmax = int(np.ceil(rr.max()))
    # bin radial distances
    bins = np.linspace(0, rmax, n_bins)
    profile = np.zeros(n_bins-1)
    for i in range(len(bins)-1):
        mask = (rr>=bins[i]) & (rr<bins[i+1])
        if mask.sum()>0:
            profile[i] = img_gray[mask].mean()
        else:
            profile[i] = 0
    # take FFT of radial profile and use magnitudes (rotation invariant)
    F = np.abs(fft(profile))
    feat = F[:nf]
    # normalize
    if feat.sum()>0:
        feat = feat/ (np.linalg.norm(feat)+1e-9)
    return np.real(feat)


def reflection_symmetry_score(img_gray):
    
    H,W = img_gray.shape
    flip = np.fliplr(img_gray)
    imgf = img_gray.astype(float)
    flipf = flip.astype(float)
    
    imgf -= imgf.mean()
    flipf -= flipf.mean()
    denom = np.sqrt((imgf**2).sum() * (flipf**2).sum()) + 1e-9
    score = (imgf*flipf).sum() / denom
    return float(score)


def extract_symmetry_invariant_features(img_rgb, resize_to=256):
    
    if img_rgb.ndim==3:
        gray = rgb2gray(img_rgb)
        gray = (gray*255).astype('uint8')
    else:
        gray = img_rgb.astype('uint8')
    gray = resize(gray, (resize_to, resize_to), preserve_range=True).astype('uint8')
    
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    hu = hu_moments(blur)
    rfd = radial_fourier_descriptor(blur)
    sym = reflection_symmetry_score(blur)
    feat = np.concatenate([hu, rfd, [sym]])
    return feat