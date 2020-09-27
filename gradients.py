import argparse
import time

from filters import *
from helpers import *

def main(args):
    # timer
    tic = time.time()

    # load images
    img = read(args.img, color=args.color)

    if args.high:
        high_img = read(args.high, args.color)

    show(img)
    # call function on each image channel
    for d in range(img.shape[2]):
        # check for special function cases
        img[:, :, d] = globals()[args.function](img[:, :, d], sigma=args.sigma, size=args.kernel_size, thresh=args.thresh)

    # show image in original format and rgb
    show(img)

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-f", "--function", default="grad_magnitude_gauss")
    ap.add_argument("-k", "--kernel_size", default=5)
    ap.add_argument("-s", "--sigma", default=2)
    ap.add_argument("-t", "--thresh", default=0.05)
    args = ap.parse_args()

    main(args)
