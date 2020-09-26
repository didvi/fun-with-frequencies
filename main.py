import argparse
import time

from filters import *
from helpers import *


def main(args):
    # timer
    tic = time.time()

    # read in the image
    img = skio.imread(args.img, as_gray=not args.color)

    # convert to double
    img = sk.img_as_float(img)

    if args.color:
        for d in range(img.shape[2]):
            img[:, :, d] = globals()[args.function](img[:, :, d])
    else:
        img = globals()[args.function](img)

    show(img)

    # show image in rgb
    show(sk.img_as_ubyte(img))

    toc = time.time()
    print("Time elapsed: " + str(toc - tic))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--img")
    ap.add_argument("-c", "--color", type=bool, default=False)
    ap.add_argument("-f", "--function", default="show")
    ap.add_argument(
        "-l",
        "--low",
        default="in/guy.jpg",
        description="Combines low frequencies of this image with high frequencies of the low image.",
    )
    ap.add_argument(
        "-h",
        "--high",
        default="in/cat.jpg",
        description="Combines high frequencies of this image with low frequencies of the low image.",
    )
    args = ap.parse_args()

    main(args)
