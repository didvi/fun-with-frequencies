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
def dxog(img, size=5, sigma=2, **kwargs):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1], [-1]], gauss)
    return convolve2d(img, dog, mode="same")


def dyog(img, size=5, sigma=2, **kwargs):
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    dog = convolve2d([[1, -1]], gauss)
    return convolve2d(img, dog, mode="same")


def dx(img, **kwargs):
    return convolve2d(img, [[1], [-1]], mode="same")


def dy(img, **kwargs):
    return convolve2d(img, [[1, -1]], mode="same")


def grad_magnitude(img, size=5, sigma=2, thresh=0.2):
    img_dx = dx(img)
    img_dy = dy(img)
    mag = np.sqrt(np.add(img_dx ** 2, img_dy ** 2))

    if thresh:
        return mag > thresh
    return mag


def grad_magnitude_gauss(img, size=5, sigma=2, thresh=0.05):
    if fast:
        img_dx = dxog(img)
        img_dy = dyog(img)
        mag = np.sqrt(np.add(img_dx ** 2, img_dy ** 2))

        if thresh:
            return mag > thresh
        return mag
    else:
        img = gauss(img)
        return grad_magnitude(img, thresh=thresh)


def grad_angle(img, size=5, sigma=2, **kwargs):
    """Calculates gradient angle for each pixel in the image

    Args:
        img (np.ndarray): image as a numpy array

    Returns:
        np.ndarray: array with the same size of the image
    """
    img_dx = dxog(img, size=size, sigma=sigma)
    img_dy = dyog(img, size=size, sigma=sigma)

    # remove junk data
    zero_mask = img_dx != 0
    img_dx = img_dx[zero_mask]
    img_dy = img_dy[zero_mask]

    zero_mask = np.sqrt(np.add(img_dx ** 2, img_dy ** 2)) > 0.05
    img_dx = img_dx[zero_mask]
    img_dy = img_dy[zero_mask]

    # convert to degrees as positive integers
    return (np.arctan(img_dy / img_dx) * 180 / np.pi).astype(int)


# FREQUENCIES
def gauss(img, size=5, sigma=2, **kwargs):
    """Convolve Image with Gaussian Filter"""
    gauss = cv2.getGaussianKernel(size, sigma) @ cv2.getGaussianKernel(size, sigma).T
    return convolve2d(img, gauss, mode="same")


def high_freq(img, size=5, sigma=2):
    """Filters out low frequencies in image

    Args:
        img (np.ndarray): input image, must be 2D
        size (int, optional): Gaussian kernel size. Defaults to 5.
        sigma (int, optional): Gaussian sigma. Defaults to 2.

    Returns:
        np.ndarray: high frequency image
    """
    low_freq = gauss(img, size=size, sigma=sigma)
    return img - low_freq


def log_filter(width, sigma, alpha):
    """Calculates the laplacian of the gaussian filter

    Args:
        width (int): kernel will be of size width x width
        sigma (double): gaussian sigma

    Returns:
        np.ndarray: returns filter
    """
    gauss = cv2.getGaussianKernel(width, sigma) @ cv2.getGaussianKernel(width, sigma).T
    unit_impulse = np.zeros((width, width))
    unit_impulse[width // 2, width // 2] = 1
    kernel = ((1 + alpha) * unit_impulse) - (alpha * gauss)

    return kernel


# TRANSFORMATIONS
def crop(img, shape):
    """Center crops the image into specified shape"""
    if img.shape == shape:
        return img

    x = img.shape[0] - shape[0]
    y = img.shape[1] - shape[1]
    return img[x // 2 : x // 2 - x, y // 2 : y // 2 - y]

def find_angle(img):
    """Finds best angle for 2D image within -10, 10 degree rotation range"""
    rotation_heuristic = np.zeros(20)
    for d in range(-10, 10):
        rotated_img = ndi.interpolation.rotate(img, d)
        rotated_img = crop(
            rotated_img, (rotated_img.shape[0] - 40, rotated_img.shape[1] - 40)
        )

        angles = grad_angle(rotated_img)
        # count all angles that = 0 mod 90
        rotation_heuristic[d + 10] = np.sum(np.mod(angles, 90) == 0)
        print(rotation_heuristic[d + 10])

    # # show histogram
    max_angle = np.argmax(rotation_heuristic) - 10
    return max_angle


def straighten(img):
    """Straightens entire image, 3d or 2d"""
    angle = find_angle(img[:, :, 0])
    print("Rotating by " + str(angle))
    img = ndi.interpolation.rotate(img, angle)
    return img


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
    sharp_img = convolve2d(img, kernel, mode="same")
    show(high_freq(img))
    return sharp_img

def combine(high_img, low_img, sigma=2, size=5):
    """Images to combine. First image contributes the high frequency, second image contributes the low frequency.

    Args:
        imgs (array): Two np arrays. Both 2D.

    Returns:
        np.ndarray: combined image
    """
    high_img = high_freq(high_img, size=size, sigma=sigma)
    low_img = gauss(low_img, size=size, sigma=sigma)
    res = np.mean(np.stack([high_img, low_img]), axis=0)
    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(high_img)))))
    plt.show()

    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(low_img)))))
    plt.show()

    plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(res)))))
    plt.show()
    return res

def gauss_laplace_stack(img, sigma=2, size=5, levels=4, show_plot=True):
    gausses = [img]
    laplaces = [img]
    prev = img
    for l in range(levels):
        img = gauss(img, size=size, sigma=sigma)
        gausses.append(img)
        laplaces.append(prev - img)
        prev = img

    if show_plot:
        show(gausses + laplaces)
    return gausses, laplaces

def left_right_mask(shape, sigma=7, size=11):
    mask = np.expand_dims(np.linspace(1, 0, num=shape[1]), axis=-1)
    mask = np.tile(mask, shape[0]).T
    left_masks, _ = gauss_laplace_stack(mask, sigma=sigma, size=size, show_plot=False)
    
    mask = np.expand_dims(np.linspace(0, 1, num=shape[1]), axis=-1)
    mask = np.tile(mask, shape[0]).T
    right_masks, _ = gauss_laplace_stack(mask, sigma=sigma, size=size, show_plot=False)
    
    return left_masks, right_masks

def strange_mask(shape, sigma=7, size=11):
    mask = np.expand_dims(np.linspace(1, 0, num=shape[1]), axis=-1)
    mask = np.tile(mask, shape[0]).T
    mask = ndi.rotate(mask, 30, mode='reflect', reshape=False)
    left_masks, _ = gauss_laplace_stack(mask, sigma=sigma, size=size, show_plot=False)
    
    mask = np.expand_dims(np.linspace(0, 1, num=shape[1]), axis=-1)
    mask = np.tile(mask, shape[0]).T
    mask = ndi.rotate(mask, 30, mode='reflect', reshape=False)
    right_masks, _ = gauss_laplace_stack(mask, sigma=sigma, size=size, show_plot=False)
    
    return left_masks, right_masks

def blend(left_img, right_img, sigma=7, size=11):
    left_masks, right_masks = strange_mask(left_img.shape, sigma=sigma, size=size)
    
    # LEFTSIDE
    _, laplaces = gauss_laplace_stack(left_img, sigma=sigma, size=size, show_plot=False)
    left = np.multiply(laplaces, left_masks)
    left = np.sum(left, axis=0)

    # # RIGHT SIDE
    _, laplaces = gauss_laplace_stack(right_img, sigma=sigma, size=size, show_plot=False)
    right = np.multiply(laplaces, right_masks)
    right = np.sum(right, axis=0)

    return np.add(left, right)



