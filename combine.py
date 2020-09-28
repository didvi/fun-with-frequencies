import argparse
import time
import os

from filters import *
from helpers import *
from align_images import align_images


def main(args):
    # timer
    tic = time.time()

    # load images
    high_img = read(args.img, color=args.color)
    low_img = read(args.low_img, color=args.color)

    high_img, low_img = align_images(high_img, low_img)
    # call function on each image channel
    for d in range(high_img.shape[2]):
        high_img[:, :, d] = combine(high_img[:, :, d], low_img[:, :, d], sigma=args.sigma, size=args.kernel_size)

    # show image in original format and rgb
    show(high_img)

    if args.save:
        save(high_img, args.low_img, sigma=args.sigma, img2=os.path.basename(args.img))

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-l", "--low_img", help="Image to combine its low frequencies with")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-s", "--sigma", type=float, default=2)
    ap.add_argument("-k", "--kernel_size", type=int, default=5)
    ap.add_argument("--save", type=bool, default=False)
    args = ap.parse_args()

    main(args)
