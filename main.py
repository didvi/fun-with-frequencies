import argparse
import time

from filters import *
from helpers import *

def main(args):
    # timer
    tic = time.time()

    # read in the image
    img = skio.imread(args.img, as_gray=True)

    # convert to double 
    img = sk.img_as_float(img)

    
    toc = time.time()
    print("Time elapsed: " + str(toc - tic))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("--method", default='exhaustive')
    ap.add_argument("--metric", default='ssd')
    ap.add_argument("--offset", type=int, default=16)
    ap.add_argument("--border", type=bool, default=False)
    ap.add_argument("--show", type=bool, default=True)
    args = ap.parse_args()

    main(args)
