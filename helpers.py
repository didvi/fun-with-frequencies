import skimage as sk
import skimage.io as skio
import numpy as np

import os 

def show(img):
    # normalize
    img = img - np.min(img)
    img = img / max(1, np.max(img))
    img = sk.img_as_ubyte(img)
    skio.imshow(img)
    skio.show()
    return img

def save(img, imname, **kwargs):
    """Saves image in images folder with kwargs in image name as {key}_{value}
    """
    # normalize and convert image
    img = img - np.min(img)
    img = img / max(1, np.max(img))
    img = sk.img_as_ubyte(img)

    fname = os.path.basename(imname)
    fname = [str(k) + '_' + str(kwargs[k]) for k in kwargs.keys()] + [fname]
    fname = "out/" + '_'.join(fname)
    skio.imsave(fname, img)
    return img
    
def read(img, color=False):
    # read in the image
    img = skio.imread(img, as_gray=not color)
    
    if not color:
        img = np.expand_dims(img, axis=2)

    # convert to double
    img = sk.img_as_float(img)
    return img