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

    # call function on each image channel
    for d in range(img.shape[2]):
        # check for special function cases
        if args.function == "combine":
            img[:, :, d] = combine(high_img[:, :, d], img[:, :, d])
        
        if args.function == 'straighten':
            img = straighten(img)
            break
        
        else:
            img[:, :, d] = globals()[args.function](img[:, :, d])

    # show image in original format and rgb
    show(img)

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-f", "--function", default="show")
    ap.add_argument(
        "--high",
        default="in/cat.jpg",
        help="Combines high frequencies of this image with blurred image. Only needed if function is combine.",
    )
    args = ap.parse_args()

    main(args)
