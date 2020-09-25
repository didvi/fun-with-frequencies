import skimage as sk
import skimage.io as skio

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
    
