import argparse

import numpy as np
import scipy as sc
from scipy import ndimage as ndi
import matplotlib.pyplot as plt 
from scipy.signal import convolve2d
import skimage.io as skio
import cv2

from helpers import *

# TODO idk if i need these
import skimage as sk
from skimage import transform
from skimage import feature

# GRADIENTS
def dxog(img, size=4, sigma=1):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1], [-1]], gauss)
    return convolve2d(img, dog, mode='same')

def dyog(img, size=4, sigma=1):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1, -1]], gauss)
    return convolve2d(img, dog, mode='same')

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

def grad_angle(img):
    """Calculates gradient angle for each pixel in the image

    Args:
        img (np.ndarray): image as a numpy array

    Returns:
        np.ndarray: array with the same size of the image
    """
    img_dx = dx(img)
    img_dy = dy(img)

    # remove junk data
    zero_mask = np.logical_and(bool_dx(img, gauss=True, thresh=0.001), bool_dy(img, gauss=True, thresh=0.001))
    img_dx = img_dx[zero_mask]
    img_dy = img_dy[zero_mask]

    # convert to degrees as positive integers
    return np.mod((np.arctan(img_dy / img_dx) * 180 / np.pi).astype(int), 360)

# FREQUENCIES
def gauss(img, size=10, sigma=4):
    """Convolve Image with Gaussian Filter
    """
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    return convolve2d(img, gauss, mode='same')

def high_freq(img, size=5, sigma=2):
    """Filters out low frequencies in image

    Args:
        img (np.ndarray): input image, must be 2D
        size (int, optional): Gaussian kernel size. Defaults to 4.
        sigma (int, optional): Gaussian sigma. Defaults to 2.

    Returns:
        np.ndarray: high frequency image
    """
    low_freq = gauss(img, size=size, sigma=sigma)
    return img - low_freq

def log_filter(width, sigma, alpha):
    # TODO fix this
    """Calculates the laplacian of the gaussian filter

    Args:
        width (int): kernel will be of size width x width
        sigma (double): gaussian sigma

    Returns:
        np.ndarray: returns filter
    """
    gauss = cv2.getGaussianKernel(width, sigma) @ cv2.getGaussianKernel(width, sigma).T
    unit_impulse = np.zeros((width, width))
    unit_impulse[width // 2, width // 2] = cv2.getGaussianKernel(1, sigma)
    kernel = ((1 + alpha) * unit_impulse) - (alpha * gauss)

    return kernel

# TRANSFORMATIONS
def crop(img, shape):
    """Center crops the image into specified shape"""
    if img.shape == shape:
        return img

    x = (img.shape[0] - shape[0])
    y = (img.shape[1] - shape[1])
    return img[x // 2:x // 2 - x, y // 2:y // 2 - y]

def rotate(img, end=False):
    # TODO fix this

    img = crop(img, (img.shape[0] - 40, img.shape[1] - 40))
    angles = grad_angle(img)
    # find maximum angle
    max_angle = np.argmax(np.bincount(angles.flatten())) - 180

    # show histogram
    plt.hist(angles)   
    plt.show()

    # rotate image
    print("Rotating by " + str(max_angle))
    rotated_img = ndi.interpolation.rotate(img, max_angle % 45)
    show(rotated_img)

def sharpen(img, sigma=2, size=5, alpha=0.5):
    """Sharpens image by alpha value

    Args:
        img (np.ndarray): input image, must be 2D
        sigma (int, optional): Gaussian blur sigma. Defaults to 1.
        alpha (float, optional): Sharpening magnitude. Defaults to 0.5.

    Returns:
        np.ndarray: sharpened image
    """
    kernel = log_filter(size, sigma, alpha=alpha)
    sharp_img = convolve2d(img, kernel, mode='same')
    return sharp_img

def align(imgs):
    # TODO
    return

def combine(high_img, low_img):
    """Images to combine. First image contributes the high frequency, second image contributes the low frequency.

    Args:
        imgs (array): Two np arrays. Both 2D.

    Returns:
        np.ndarray: combined image
    """
    high_img = high_freq(high_img)
    low_img = gauss(low_img)

    shape = (min(high_img.shape[0], low_img.shape[0]), min(high_img.shape[1], low_img.shape[1]))

    high_img = crop(high_img, shape=shape)
    low_img = crop(low_img, shape=shape)
    return high_img + low_img