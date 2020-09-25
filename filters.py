import argparse

import numpy as np
import scipy as sc
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
import skimage.io as skio
import cv2

from helpers import *

# TODO idk if i need these
import skimage as sk
from skimage import transform
from skimage import feature
from scipy import ndimage as ndi

def gauss(img, size=4, sigma=2):
    """Convolve Image with Gaussian Filter
    """
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    return convolve2d(img, gauss)

def dxog(img, size=4, sigma=2, thresh=0.05):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1], [-1]], gauss)
    return np.abs(convolve2d(img, dog)) > thresh

def dyog(img, size=4, sigma=2, thresh=0.05):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1, -1]], gauss)
    return np.abs(convolve2d(img, dog)) > thresh

def dx(img):
    return convolve2d(img, [[1], [-1]], mode='same')

def dy(img):
    return convolve2d(img, [[1, -1]], mode='same')

def bool_dx(img, gauss=False, thresh=0.5):
    if gauss:
        return np.abs(dxog(img)) > thresh
    return np.abs(dx(img)) > thresh

def bool_dy(img, gauss=False, thresh=0.5):
    if gauss:
        return np.abs(dyog(img)) > thresh
    return np.abs(dy(img)) > thresh

def crop(img, shape):
    """Center crops the image into specified shape"""
    x = (img.shape[0] - shape[0]) // 2
    y = (img.shape[1] - shape[1]) // 2
    return img[x:-x, y:-y]

def grad_angle(img):
    """Calculates gradient angle for each pixel in the image

    Args:
        img (np.ndarray): image as a numpy array

    Returns:
        np.ndarray: array with the same size of the image
    """
    img_dx = dx(img)
    img_dy = dy(img)
    return np.round(np.degrees(np.arctan(img_dy / img_dx)))

def rotate(img):
    angles = grad_angle(img)
    show(angles)
    # plt.plot(angles)

    plt.show()