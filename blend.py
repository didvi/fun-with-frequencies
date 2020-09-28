import argparse
import time
import os

from filters import *
from helpers import *
from align_images import *


def main(args):
    # timer
    tic = time.time()

    # load images
    img1 = read(args.img, color=args.color)
    img2 = read(args.img2, color=args.color)

    img1, img2 = match_img_size(img1, img2)

    # call function on each image channel
    for d in range(img1.shape[2]):
        img1[:, :, d] = blend(img1[:, :, d], img2[:, :, d], sigma=args.sigma, size=args.kernel_size)

    # show image in original format and rgb
    show(img1)

    if args.save:
        save(img1, args.img2, sigma=args.sigma, img2=os.path.basename(args.img), function="blend")

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("--img2", help="Image to blend with")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-s", "--sigma", type=float, default=7)
    ap.add_argument("-k", "--kernel_size", type=int, default=11)
    ap.add_argument("--save", type=bool, default=False)
    args = ap.parse_args()

    main(args)
