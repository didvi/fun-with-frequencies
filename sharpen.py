import argparse
import time

from filters import *
from helpers import *

def main(args):
    # timer
    tic = time.time()

    # load images
    img = read(args.img, color=args.color)

    show(img)
    # call function on each image channel
    for d in range(img.shape[2]):
        img[:, :, d] = sharpen(img[:, :, d], sigma=args.sigma, size=args.kernel_size, alpha=args.alpha)

    # show image in original format and rgb
    show(img)

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-a", "--alpha", type=float, default=0.5)
    ap.add_argument("-s", "--sigma", type=float, default=2)
    ap.add_argument("-k", "--kernel_size", type=int, default=5)
    args = ap.parse_args()

    main(args)
