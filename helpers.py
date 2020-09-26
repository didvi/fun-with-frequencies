import skimage as sk
import skimage.io as skio
import numpy as np

import os 

def show(img):
    skio.imshow(img)
    skio.show()

def save(img, imname, **kwargs):
    """Saves image in images folder with kwargs in image name as {key}_{value}
    """
    fname = os.path.basename(imname)
    fname = [str(k) + '_' + str(kwargs[k]) for k in kwargs.keys()] + [fname]
    fname = "out/" + '_'.join(fname)
    skio.imsave(fname, img)
    
def read(img, color=False):
    # read in the image
    img = skio.imread(img, as_gray=not color)
    
    if not color:
        img = np.expand_dims(img, axis=2)

    # convert to double
    img = sk.img_as_float(img)
    return img