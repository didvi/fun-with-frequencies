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
def dxog(img, size=5, sigma=2):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1], [-1]], gauss)
    return convolve2d(img, dog, mode='same')

def dyog(img, size=5, sigma=2):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1, -1]], gauss)
    return convolve2d(img, dog, mode='same')

def dx(img):
    return convolve2d(img, [[1], [-1]], mode='same')

def dy(img):
    return convolve2d(img, [[1, -1]], mode='same')

def grad_magnitude(img, thresh=0.2):
    img_dx = dx(img)
    img_dy = dy(img)
    mag = np.sqrt(np.add(img_dx**2, img_dy**2))

    if thresh:
        return mag > thresh
    return mag

def grad_magnitude_gauss(img, thresh=0.05, fast=True):
    if fast:
        img_dx = dxog(img)
        img_dy = dyog(img)
        mag = np.sqrt(np.add(img_dx**2, img_dy**2))

        if thresh:
            return mag > thresh
        return mag
    else:
        img = gauss(img)
        return grad_magnitude(img, thresh=thresh)

def grad_angle(img):
    """Calculates gradient angle for each pixel in the image

    Args:
        img (np.ndarray): image as a numpy array

    Returns:
        np.ndarray: array with the same size of the image
    """
    img_dx = dxog(img)
    img_dy = dyog(img)

    # remove junk data
    zero_mask = img_dx != 0
    img_dx = img_dx[zero_mask]
    img_dy = img_dy[zero_mask]

    # convert to degrees as positive integers
    return (np.arctan(img_dy / img_dx) * 180 / np.pi).astype(int)

# FREQUENCIES
def gauss(img, size=5, sigma=2):
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

def find_angle(img):
    """Finds best angle for 2D image within -10, 10 degree rotation range"""
    rotation_heuristic = np.zeros(20)
    for d in range(-10, 10):
        rotated_img = ndi.interpolation.rotate(img, d)
        rotated_img = crop(rotated_img, (rotated_img.shape[0] - 40, rotated_img.shape[1] - 40))

        angles = grad_angle(rotated_img)
        # count all angles that = 0 mod 90
        rotation_heuristic[d + 10] = np.sum(np.mod(angles, 90) == 0)
        print(rotation_heuristic[d + 10])

    # # show histogram
    # plt.hist(angles)   
    max_angle = np.argmax(rotation_heuristic) - 10
    return max_angle

def straighten(img):
    """Straightens entire image, 3d or 2d"""   
    angle = find_angle(img[:, :, 0])
    print("Rotating by " + str(angle))
    return ndi.interpolation.rotate(img, angle)

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