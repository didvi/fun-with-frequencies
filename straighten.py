import argparse
import time

from filters import *
from helpers import *

def main(args):
    # timer
    tic = time.time()

    # load images
    img = read(args.img, color=args.color)

    # show before image
    show(img)

    img = straighten(img)

    # show after image
    show(img)

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-c", "--color", type=bool, default=True)
    args = ap.parse_args()

    main(args)
